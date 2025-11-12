//! Coin Grinder.
//!
//! This module introduces the Coin Grinder selection algorithm.
//!

use crate::{
    types::{CoinSelectionOpt, OutputGroup, SelectionError, SelectionOutput},
    utils::{calculate_fee, calculate_waste, sum},
};

const ITERATION_LIMIT: u32 = 100_000;

/// The sum of UTXO amounts after this UTXO index, e.g. lookahead[5] = Σ(UTXO[6+].amount)
fn build_lookahead(
    feerate: f32,
    lookahead: Vec<(usize, &OutputGroup)>,
    available_value: u64,
) -> Result<Vec<u64>, SelectionError> {
    Ok(lookahead
        .iter()
        .try_fold(Vec::new(), |mut acc, u| {
            acc.push(u.1.effective_value(feerate)?);
            Ok(acc)
        })?
        .iter()
        .scan(available_value, |state, u| {
            *state -= u;
            Some(*state)
        })
        .collect())
}

// Provides a lookup to determine the minimum UTXO weight after a given index.
fn build_min_tail_weight(weighted_utxos: Vec<(usize, &OutputGroup)>) -> Vec<u64> {
    let weights: Vec<_> = weighted_utxos
        .into_iter()
        .map(|u| u.1.weight)
        .rev()
        .collect();
    let mut prev = u64::MAX;
    let mut result = Vec::new();
    for w in weights {
        result.push(prev);
        prev = std::cmp::min(prev, w);
    }
    result.into_iter().rev().collect()
}

/// From the indices of the sorted list, we find the original indexes given to the algo
fn index_to_original_indices(
    index_list: Vec<usize>,
    wu: Vec<(usize, &OutputGroup)>,
) -> Result<Vec<(usize, &OutputGroup)>, SelectionError> {
    let mut result = Vec::new();

    for i in index_list {
        let wu = wu[i];
        result.push(wu);
    }

    Ok(result)
}

/// check errors before sending the result
fn result_check(
    results: &[usize],
    iterations: u32,
    max_tx_weight_exceeded: bool,
) -> Result<(), SelectionError> {
    if results.is_empty() {
        if iterations == ITERATION_LIMIT {
            Err(SelectionError::IterationLimitReached)
        } else if max_tx_weight_exceeded {
            Err(SelectionError::MaxWeightExceeded)
        } else {
            Err(SelectionError::NoSolutionFound)
        }
    } else {
        Ok(())
    }
}

fn result(
    options: &CoinSelectionOpt,
    best_selection: Vec<usize>,
    weighted_utxos: Vec<(usize, &OutputGroup)>,
    iteration: u32,
    max_tx_weight_exceeded: bool,
) -> Result<SelectionOutput, SelectionError> {
    let selected_inputs = index_to_original_indices(best_selection, weighted_utxos)?;
    let selected_indices: Vec<usize> = selected_inputs.iter().map(|u| u.0).collect();
    result_check(&selected_indices, iteration, max_tx_weight_exceeded)?;
    let selected_output: Vec<&OutputGroup> = selected_inputs.iter().map(|u| u.1).collect();
    let accumulated_weight = selected_output.iter().fold(0, |acc, x| acc + x.weight);
    let estimated_fee = calculate_fee(accumulated_weight, options.target_feerate)?;
    // the accumulated value != effective value
    let accumulated_value = selected_output.iter().fold(0, |acc, x| acc + x.value);
    let waste = crate::types::WasteMetric(calculate_waste(
        options,
        accumulated_value,
        accumulated_weight,
        estimated_fee,
    )?);
    Ok(SelectionOutput {
        selected_inputs: selected_indices,
        waste,
        iterations: iteration,
    })
}

// Estimate if any combination of remaining inputs would be higher than `best_weight`
fn is_remaining_weight_higher(
    weight_total: u64,
    min_tail_weight: u64,
    target: u64,
    amount_total: u64,
    tail_amount: u64,
    best_weight: u64,
) -> Option<bool> {
    // amount remaining until the target is reached.
    let remaining_amount = target.checked_sub(amount_total)?;

    // number of inputs left to reach the target.
    // TODO use checked div rounding up
    let utxo_count = remaining_amount.div_ceil(tail_amount);

    // sum of input weights if all inputs are the best possible weight.
    let remaining_weight = min_tail_weight * utxo_count;

    // add remaining_weight to the current running weight total.
    let best_possible_weight = weight_total + remaining_weight;

    Some(best_possible_weight > best_weight)
}

/// Deterministic Branch and Bound search that minimizes the input weight.
///
/// This algorithm selects the set of inputs that meets the `total_target` and has the lowest
/// total weight.  In so doing, a change output is created unlike the vanilla Branch and Bound
/// algorithm.  Therefore, in order to ensure that the change output can be paid for, the
/// `total_target` is calculated as `target` plus `change_target` where `change_target`.  The
/// `change_target` is the budgeted amount to pay for the change output.
///
/// See also: [bitcoin coin selection](https://github.com/bitcoin/bitcoin/blob/62bd61de110b057cbfd6e31e4d0b727d93119c72/src/wallet/coinselection.cpp#L204)
///
/// There is discussion [here](https://murch.one/erhardt2016coinselection.pdf) at section 6.4.3
/// that prioritizing input weight will lead to a fragmentation of the UTXO set.  Therefore, prefer
/// this search only in extreme conditions where fee_rate is high, since a set of UTXOs with minimal
/// weight will lead to a cheaper constructed transaction in the short term.  However, in the
/// long-term, this prioritization can lead to more UTXOs to choose from.
pub fn coin_grinder(
    weighted_utxos: &[OutputGroup],
    options: &CoinSelectionOpt,
) -> Result<SelectionOutput, SelectionError> {
    weighted_utxos
        .iter()
        .map(|u| u.weight)
        .try_fold(0, u64::checked_add)
        .ok_or(SelectionError::AbnormallyHighAmount)?;

    let available_value = weighted_utxos
        .iter()
        .try_fold(Vec::new(), |mut acc, u| {
            acc.push(u.effective_value(options.target_feerate)?);
            Ok(acc)
        })?
        .into_iter()
        .try_fold(0, u64::checked_add)
        .ok_or(SelectionError::AbnormallyHighAmount)?;

    let mut weighted_utxos: Vec<(usize, &OutputGroup)> =
        weighted_utxos.iter().enumerate().collect();
    OutputGroup::sort_by_effective_value(&mut weighted_utxos, options.target_feerate)?;

    let lookahead = build_lookahead(
        options.target_feerate,
        weighted_utxos.clone(),
        available_value,
    )?;
    let min_tail_weight = build_min_tail_weight(weighted_utxos.clone());

    let total_target = sum(options.target_value, options.min_change_value)?;

    if available_value < total_target {
        return Err(SelectionError::InsufficientFunds);
    }

    if weighted_utxos.is_empty() {
        return Err(SelectionError::NoSolutionFound);
    }

    let mut selection: Vec<usize> = vec![];
    let mut best_selection: Vec<usize> = vec![];

    let mut amount_total: u64 = u64::MIN;
    let mut best_amount: u64 = u64::MAX;

    let mut weight_total = 0;
    let mut best_weight = options.max_selection_weight;
    let mut max_tx_weight_exceeded = false;

    let mut next_utxo_index = 0;
    let mut iteration: u32 = 0;

    loop {
        // Given a target of 11, and candidate set: [10/2, 7/1, 5/1, 4/2]
        //
        //      o
        //     /
        //  10/2
        //   /
        // 17/3
        //
        // A solution 17/3 is recorded, however the total of 11 is exceeded.
        // Therefor, 7/1 is shifted to the exclusion branch and 5/1 is added.
        //
        //      o
        //     / \
        //  10/2
        //   / \
        //   17/3
        //    /
        //  15/3
        //
        // This operation happens when "shift" is true.  That is, move from
        // the inclusion branch 17/3 via the omission branch 10/2 to it's
        // inclusion-branch child 15/3
        let mut shift = false;

        // Given a target of 11, and candidate set: [10/2, 7/1, 5/1, 4/2]
        // Solutions, 17/3 (shift) 15/3 (shift) and 14/4 are evaluated.
        //
        // At this point, the leaf node 14/4 makes a shift impossible
        // since there is not an inclusion-branch child.  In other words,
        // this is a leaf node.
        //
        //      o
        //     /
        //  10/2
        //    \
        //     \
        //     /
        //   14/4
        //
        // Instead we go to the omission branch of the nodes last ancestor.
        // That is, we "cut" removing every child of 10/2 and shift 10/2
        // to the omission branch.
        //
        //      o
        //     / \
        //      10/2
        let mut cut = false;

        let utxo = &weighted_utxos[next_utxo_index];
        let eff_value = utxo.1.effective_value(options.target_feerate)?;

        amount_total = sum(amount_total, eff_value)?;
        weight_total += utxo.1.weight;

        selection.push(next_utxo_index);
        next_utxo_index += 1;
        iteration += 1;

        let tail: usize = *selection
            .last()
            .expect("we just added an element, so last should always be Some");
        if sum(amount_total, lookahead[tail])? < total_target {
            cut = true;
        } else if weight_total > best_weight {
            max_tx_weight_exceeded = true;
            if weighted_utxos[tail].1.weight <= min_tail_weight[tail] {
                cut = true;
            } else {
                shift = true;
            }
        } else if amount_total >= total_target {
            shift = true;
            if weight_total < best_weight
                || weight_total == best_weight && amount_total < best_amount
            {
                best_selection = selection.clone();
                best_weight = weight_total;
                best_amount = amount_total;
            }
        } else if !best_selection.is_empty() {
            if let Some(is_higher) = is_remaining_weight_higher(
                weight_total,
                min_tail_weight[tail],
                total_target,
                amount_total,
                weighted_utxos[tail]
                    .1
                    .effective_value(options.target_feerate)?,
                best_weight,
            ) {
                if is_higher {
                    if weighted_utxos[tail].1.weight <= min_tail_weight[tail] {
                        cut = true;
                    } else {
                        shift = true;
                    }
                }
            }
        }

        if iteration >= ITERATION_LIMIT {
            return result(
                options,
                best_selection,
                weighted_utxos,
                iteration,
                max_tx_weight_exceeded,
            );
        }

        // check if evaluating a leaf node.
        if next_utxo_index == weighted_utxos.len() {
            cut = true;
        }

        if cut {
            // deselect
            let utxo = weighted_utxos[*selection.last().unwrap()];
            let eff_value = utxo.1.effective_value(options.target_feerate)?;

            amount_total = amount_total
                .checked_sub(eff_value)
                .ok_or(SelectionError::AbnormallyHighAmount)?;
            weight_total -= utxo.1.weight;
            selection.pop();
            shift = true;
        }

        while shift {
            if selection.is_empty() {
                return result(
                    options,
                    best_selection,
                    weighted_utxos,
                    iteration,
                    max_tx_weight_exceeded,
                );
            }

            next_utxo_index = selection.last().unwrap() + 1;

            // deselect
            let utxo = weighted_utxos[*selection.last().unwrap()];
            let eff_value = utxo.1.effective_value(options.target_feerate)?;

            amount_total = amount_total
                .checked_sub(eff_value)
                .ok_or(SelectionError::AbnormallyHighAmount)?;
            weight_total -= utxo.1.weight;
            selection.pop();

            shift = false;

            // skip all next inputs that are equivalent to the current input
            // if the current input didn't contribute to a solution.
            while weighted_utxos[next_utxo_index - 1]
                .1
                .effective_value(options.target_feerate)?
                == weighted_utxos[next_utxo_index]
                    .1
                    .effective_value(options.target_feerate)?
            {
                if next_utxo_index >= weighted_utxos.len() - 1 {
                    shift = true;
                    break;
                }
                next_utxo_index += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[derive(Debug)]
    pub struct TestCoinGrinder<'a> {
        options: CoinSelectionOpt,
        inputs: &'a [OutputGroup],
        expected_utxos: Vec<usize>,
        expected_error: Option<SelectionError>,
        expected_iterations: u32,
    }

    impl TestCoinGrinder<'_> {
        fn assert(&self) {
            let result = coin_grinder(self.inputs, &self.options);

            match result {
                Ok(selection_output) => {
                    assert_eq!(selection_output.iterations, self.expected_iterations);
                    assert_eq!(selection_output.selected_inputs, self.expected_utxos);
                }
                Err(e) => {
                    let expected_error = self.expected_error.unwrap();
                    assert!(self.expected_utxos.is_empty());
                    assert_eq!(e, expected_error);
                }
            }
        }
    }

    #[test]
    fn min_tail_weight() {
        let wu = &[
            OutputGroup::new(29, 36),
            OutputGroup::new(19, 40),
            OutputGroup::new(11, 44),
        ];
        let enumerated_wu = wu.iter().enumerate().collect();

        let min_tail_weight = build_min_tail_weight(enumerated_wu);

        let expect: Vec<u64> = vec![40, 44, 18446744073709551615];
        assert_eq!(min_tail_weight, expect);
    }

    #[test]
    fn lookahead() {
        let utxos = [
            OutputGroup::new(10, 8),
            OutputGroup::new(7, 4),
            OutputGroup::new(5, 4),
            OutputGroup::new(4, 8),
        ];
        let lookahead = build_lookahead(0.0, utxos.iter().enumerate().collect(), 26).unwrap();

        let expect: Vec<u64> = vec![16, 9, 4, 0];

        assert_eq!(lookahead, expect);
    }

    #[test]
    fn example_solution() {
        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 11,
                target_feerate: 0.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 0,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 100,
            },
            inputs: &[
                OutputGroup::new(10, 8),
                OutputGroup::new(7, 4),
                OutputGroup::new(5, 4),
                OutputGroup::new(4, 8),
            ],
            expected_utxos: vec![1, 2],
            expected_error: None,
            expected_iterations: 8,
        }
        .assert();
    }

    #[test]
    fn insufficient_funds() {
        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 4_950_000_000,
                target_feerate: 0.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 10_000,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: u64::MAX,
            },
            inputs: &[
                OutputGroup::new(100_000_000, 272),
                OutputGroup::new(200_000_000, 272),
            ],
            expected_utxos: vec![],
            expected_error: Some(SelectionError::InsufficientFunds),
            expected_iterations: 0,
        }
        .assert();
    }

    #[test]
    fn max_weight_exceeded() {
        // 2) Test max weight exceeded
        // https://github.com/bitcoin/bitcoin/blob/43e71f74988b2ad87e4bfc0e1b5c921ab86ec176/src/wallet/test/coinselector_tests.cpp#L1153
        let mut wu = Vec::new();
        for _i in 0..10 {
            wu.push(OutputGroup::new(100_000_000, 272));
            wu.push(OutputGroup::new(200_000_000, 272));
        }

        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 2_950_000_000,
                target_feerate: 5.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 1_000_000,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 3000,
            },
            inputs: &wu[..],
            expected_utxos: vec![],
            expected_error: Some(SelectionError::MaxWeightExceeded),
            expected_iterations: 0,
        }
        .assert();
    }

    #[test]
    fn max_weight_with_result() {
        // 3) Test that the lowest-weight solution is found when some combinations would exceed the allowed weight
        // https://github.com/bitcoin/bitcoin/blob/43e71f74988b2ad87e4bfc0e1b5c921ab86ec176/src/wallet/test/coinselector_tests.cpp#L1171
        let mut wu = Vec::new();
        let mut expected: Vec<usize> = Vec::new();

        for _i in 0..60 {
            wu.push(OutputGroup::new(33_000_000, 272));
        }
        for _i in 0..10 {
            wu.push(OutputGroup::new(200_000_000, 272));
        }

        for i in 0..10 {
            expected.push(60 + i);
        }
        for i in 0..17 {
            expected.push(i);
        }

        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 2_533_000_000,
                target_feerate: 5.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 1_000_000,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 10_000,
            },
            inputs: &wu[..],
            expected_utxos: expected,
            expected_error: None,
            expected_iterations: 37,
        }
        .assert();
    }

    #[test]
    fn select_lighter_utxos() {
        // 4) Test that two less valuable UTXOs with a combined lower weight are preferred over a more valuable heavier UTXO
        // https://github.com/bitcoin/bitcoin/blob/43e71f74988b2ad87e4bfc0e1b5c921ab86ec176/src/wallet/test/coinselector_tests.cpp#L1193
        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 190_000_000,
                target_feerate: 5.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 1_000_000,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 400_000,
            },
            inputs: &[
                OutputGroup::new(200_000_000, 592),
                OutputGroup::new(100_000_000, 272),
                OutputGroup::new(100_000_000, 272),
            ],
            expected_utxos: vec![1, 2],
            expected_error: None,
            expected_iterations: 3,
        }
        .assert();
    }

    #[test]
    fn select_best_weight() {
        // 5) Test finding a solution in a UTXO pool with mixed weights
        // https://github.com/bitcoin/bitcoin/blob/43e71f74988b2ad87e4bfc0e1b5c921ab86ec176/src/wallet/test/coinselector_tests.cpp#L1215
        let wu = &[
            OutputGroup::new(100_000_000, 600),
            OutputGroup::new(200_000_000, 1000),
            OutputGroup::new(300_000_000, 1400),
            OutputGroup::new(400_000_000, 600),
            OutputGroup::new(500_000_000, 1000),
            OutputGroup::new(600_000_000, 1400),
            OutputGroup::new(700_000_000, 600),
            OutputGroup::new(800_000_000, 1000),
            OutputGroup::new(900_000_000, 1400),
            OutputGroup::new(1_000_000_000, 600),
            OutputGroup::new(1_100_000_000, 1000),
            OutputGroup::new(1_200_000_000, 1400),
            OutputGroup::new(1_300_000_000, 600),
            OutputGroup::new(1_400_000_000, 1000),
            OutputGroup::new(1_500_000_000, 1400),
        ];

        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 3_000_000_000,
                target_feerate: 5.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 1_000_000,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 400_000,
            },
            inputs: &wu[..],
            expected_utxos: vec![13, 12, 3],
            expected_error: None,
            expected_iterations: 92,
        }
        .assert();
    }

    #[test]
    fn lightest_among_many_clones() {
        // 6) Test that the lightest solution among many clones is found
        // https://github.com/bitcoin/bitcoin/blob/43e71f74988b2ad87e4bfc0e1b5c921ab86ec176/src/wallet/test/coinselector_tests.cpp#L1244
        let mut wu = vec![
            OutputGroup::new(400_000_000, 400),
            OutputGroup::new(300_000_000, 400),
            OutputGroup::new(200_000_000, 400),
            OutputGroup::new(100_000_000, 400),
        ];

        for _i in 0..100 {
            wu.push(OutputGroup::new(800_000_000, 4000));
            wu.push(OutputGroup::new(700_000_000, 3200));
            wu.push(OutputGroup::new(600_000_000, 2400));
            wu.push(OutputGroup::new(500_000_000, 1600));
        }

        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 989_999_999,
                target_feerate: 5.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 1_000_000,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 400_000,
            },
            inputs: &wu[..],
            expected_utxos: vec![0, 1, 2, 3],
            expected_error: None,
            expected_iterations: 38,
        }
        .assert();
    }

    #[test]
    fn skip_tiny_inputs() {
        // 7) Test that lots of tiny UTXOs can be skipped if they are too heavy while there are enough funds in lookahead
        // https://github.com/bitcoin/bitcoin/blob/43e71f74988b2ad87e4bfc0e1b5c921ab86ec176/src/wallet/test/coinselector_tests.cpp#L1283
        // BTC core impl will return a result even if the max limit of iteration is reached
        // TODO: see if we also need to return the result, add a warning or if the waste is sufficient
        let mut wu = vec![
            OutputGroup::new(180_000_000, 10_000),
            OutputGroup::new(100_000_000, 4000),
            OutputGroup::new(100_000_000, 4000),
        ];
        let mut tiny = vec![];
        for i in 0..100 {
            tiny.push(0.01 * 100000000_f64 + i as f64);
        }
        let mut tiny: Vec<OutputGroup> = tiny
            .iter()
            .map(|a| OutputGroup::new(*a as u64, 440))
            .collect();
        wu.append(&mut tiny);

        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 190_000_000,
                target_feerate: 5.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 1_000_000,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 400_000,
            },
            inputs: &wu[..],
            expected_utxos: vec![1, 2],
            expected_error: Some(SelectionError::IterationLimitReached),
            expected_iterations: 7,
        }
        .assert();
    }

    #[test]
    fn coins_with_max_weight_does_not_overflow() {
        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 11,
                target_feerate: 0.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 0,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 100,
            },
            inputs: &[
                OutputGroup::new(10, u64::MAX),
                OutputGroup::new(7, 4),
                OutputGroup::new(5, 4),
                OutputGroup::new(4, u64::MAX),
            ],
            expected_utxos: vec![],
            expected_error: Some(SelectionError::AbnormallyHighAmount),
            expected_iterations: 8,
        }
        .assert();
    }

    #[test]
    fn max_target_and_max_change_target() {
        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: u64::MAX,
                target_feerate: 0.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: u64::MAX,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 100,
            },
            inputs: &[
                OutputGroup::new(10, 8),
                OutputGroup::new(7, 4),
                OutputGroup::new(5, 4),
                OutputGroup::new(4, 8),
            ],
            expected_utxos: vec![],
            expected_error: Some(SelectionError::AbnormallyHighAmount),
            expected_iterations: 8,
        }
        .assert();
    }

    #[test]
    fn no_available_value() {
        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 0,
                target_feerate: 0.0,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 0,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 0,
            },
            inputs: &[],
            expected_utxos: vec![],
            expected_error: Some(SelectionError::NoSolutionFound),
            expected_iterations: 0,
        }
        .assert();
    }

    #[test]
    fn effective_value_tie() {
        // A secondary sort by weight will evaluate the lightest UTXOs first when effective_value
        // is a tie.
        TestCoinGrinder {
            options: CoinSelectionOpt {
                target_value: 1500,
                target_feerate: 0.01,
                long_term_feerate: None,
                min_absolute_fee: 0,
                base_weight: 0,
                change_weight: 0,
                change_cost: 0,
                avg_input_weight: 0,
                avg_output_weight: 0,
                min_change_value: 100,
                excess_strategy: crate::types::ExcessStrategy::ToChange,
                max_selection_weight: 1000,
            },
            inputs: &[
                OutputGroup::new(1006, 592),
                OutputGroup::new(1003, 272),
                OutputGroup::new(1003, 272),
            ],
            expected_utxos: vec![1, 2],
            expected_error: None,
            expected_iterations: 2,
        }
        .assert();
    }

    // This tests that coin-grinder will find the lowest weight solution.  To do so, create two
    // random UTXOs sets, one of which has weight greater than zero, the other with weights
    // equal to zero.  Then merge the two sets and assert coin-grinder finds the solution with
    // the zero weight UTXOs.
    #[proptest]
    fn coin_grinder_proptest_lowest_weight_solution(
        exclusion_set: Vec<OutputGroup>,
        inclusion_set: Vec<OutputGroup>,
    ) {
        let mut rng = rand::thread_rng();
        let mut weight_pool: Vec<OutputGroup> = exclusion_set
            .iter()
            .filter_map(|utxo| {
                let weight = rng.gen_range(1..u64::MAX);
                Some(OutputGroup::new(utxo.value, weight))
            })
            .collect();

        let mut weightless_pool: Vec<OutputGroup> = inclusion_set
            .iter()
            .map(|utxo| OutputGroup::new(utxo.value, 0))
            .filter(|utxo| utxo.value == 0)
            .collect();
        if let Some(target) = weightless_pool
            .iter()
            .map(|utxo| utxo.value)
            .try_fold(0, u64::checked_add)
        {
            if !weightless_pool.is_empty() {
                weightless_pool.sort_by(|a, b| b.value.cmp(&a.value).then(b.weight.cmp(&a.weight)));
                weight_pool.append(&mut weightless_pool.clone());
                if weight_pool
                    .iter()
                    .map(|utxo| utxo.value)
                    .try_fold(0, u64::checked_add)
                    .is_some()
                {
                    let weight_sum = weight_pool
                        .iter()
                        .try_fold(0u64, |acc, itm| acc.checked_add(itm.weight));
                    if weight_sum.is_some() {
                        let change_target = 0u64;
                        let max_selection_weight = u64::MAX;
                        let opts = CoinSelectionOpt {
                            target_value: target,
                            min_change_value: change_target,
                            max_selection_weight,
                            ..Default::default()
                        };
                        let selection_output = coin_grinder(&weight_pool, &opts).unwrap();
                        let utxos: Vec<OutputGroup> = selection_output
                            .selected_inputs
                            .iter()
                            .filter_map(|i| weight_pool.get(*i).cloned())
                            .collect();
                        prop_assert_eq!(weightless_pool, utxos);
                        prop_assert!(selection_output.iterations > 0);
                    }
                }
            }
        }
    }

    use proptest::{num::u64, prop_assert, prop_assert_eq};
    use rand::Rng;
    use test_strategy::proptest;
    #[proptest]
    fn coin_grinder_proptest_any_solution(inputs: Vec<OutputGroup>, opts: CoinSelectionOpt) {
        let result = coin_grinder(&inputs, &opts);
        match result {
            Ok(r) => {
                // check that at least one iteration occured
                prop_assert!(r.iterations > 0);
            }
            Err(SelectionError::AbnormallyHighAmount) => {
                let available_value = inputs
                    .iter()
                    .filter_map(|u| u.effective_value(opts.target_feerate).ok())
                    .try_fold(0u64, |acc, u| acc.checked_add(u));
                let weight_total = inputs
                    .iter()
                    .map(|u| u.weight)
                    .try_fold(0u64, |acc, u| acc.checked_add(u));
                prop_assert!(
                    available_value.is_none()
                        || weight_total.is_none()
                        || opts
                            .target_value
                            .checked_add(opts.min_change_value)
                            .is_none()
                );
            }
            Err(SelectionError::InsufficientFunds) => {
                let available_value = inputs
                    .iter()
                    .filter_map(|u| u.effective_value(opts.target_feerate).ok())
                    .try_fold(0u64, |acc, u| acc.checked_add(u))
                    .unwrap_or(u64::MAX);
                prop_assert!(
                    available_value
                        < (opts
                            .target_value
                            .checked_add(opts.min_change_value)
                            .unwrap_or(u64::MAX)
                            .checked_add(opts.change_cost)
                            .unwrap_or(u64::MAX))
                );
            }
            Err(SelectionError::MaxWeightExceeded) => {
                let weight_total = inputs
                    .iter()
                    .map(|u| u.weight)
                    .try_fold(0u64, |acc, u| acc.checked_add(u))
                    .unwrap_or_default();
                prop_assert!(weight_total > opts.max_selection_weight);
            }
            Err(SelectionError::NoSolutionFound) => {
                prop_assert!(inputs.is_empty() || opts.target_value == 0)
            }
            _ => {}
        }
    }
}
