use crate::{
    types::{CoinSelectionOpt, OutputGroup, SelectionError, SelectionOutput, WasteMetric},
    utils::{available_value, calculate_fee, calculate_waste, sum},
};

// Total_Tries in Core:
// https://github.com/bitcoin/bitcoin/blob/1d9da8da309d1dbf9aef15eb8dc43b4a2dc3d309/src/wallet/coinselection.cpp#L74
pub const ITERATION_LIMIT: u32 = 100_000;

pub fn select_coin_bnb(
    inputs: &[OutputGroup],
    options: &CoinSelectionOpt,
    // ) -> Return<'a> {
) -> Result<SelectionOutput, SelectionError> {
    let mut iteration = 0;
    let mut index = 0;
    let mut max_tx_weight_exceeded = false;
    let mut backtrack;

    let mut value = 0;
    let mut weight = 0;

    let mut current_waste = 0i64;
    // cast ok, MAX_MONEY < i64::MAX
    let mut best_waste = options.max_value as i64;

    let mut index_selection: Vec<usize> = vec![];
    let mut best_selection: Vec<usize> = vec![];

    let upper_bound = sum(options.target_value, options.change_cost)?;

    let mut available_value: u64 = available_value(inputs, options.target_feerate)?;
    dbg!(&available_value);
    dbg!(&options.target_value);

    let mut weighted_utxos: Vec<_> = inputs.iter().collect();
    let long_term_feerate = if let Some(ltf) = options.long_term_feerate {
        ltf
    } else {
        return Err(SelectionError::LongTermFeeRateMissing);
    };

    let total_value = inputs.iter().try_fold(0, |acc, item| {
        Ok(acc + item.effective_value(options.target_feerate)?)
    })?;
    let total_weight: u64 = inputs.iter().map(|i| i.weight).sum();
    let estimated_fees = calculate_fee(total_weight, options.target_feerate).unwrap_or(0);
    let waste = calculate_waste(options, total_value, total_weight, estimated_fees)?;

    // descending sort by effective_value, ascending sort by waste.
    weighted_utxos.sort_by(|a, b| {
        b.effective_value(options.target_feerate)
            .cmp(&a.effective_value(options.target_feerate))
            .then(
                (((a.weight as f32 * options.target_feerate) - long_term_feerate) as u64).cmp(
                    &(((b.weight as f32 * options.target_feerate) - long_term_feerate) as u64),
                ),
            )
    });

    if available_value < options.target_value {
        return Err(SelectionError::InsufficientFunds);
    }

    while iteration < ITERATION_LIMIT {
        backtrack = false;
        dbg!(iteration);
        dbg!(&value);
        dbg!(&value);

        // * If any of the conditions are met, backtrack.
        //
        // unchecked_add is used here for performance.  Before entering the search loop, all
        // utxos are summed and checked for overflow.  Since there was no overflow then, any
        // subset of addition will not overflow.
        if available_value + value < options.target_value
            // Provides an upper bound on the excess value that is permissible.
            // Since value is lost when we create a change output due to increasing the size of the
            // transaction by an output (the change output), we accept solutions that may be
            // larger than the target.  The excess is added to the solutions waste score.
            // However, values greater than value + cost_of_change are not considered.
            //
            // This creates a range of possible solutions where;
            // range = (target, target + cost_of_change]
            //
            // That is, the range includes solutions that exactly equal the target up to but not
            // including values greater than target + cost_of_change.
            || value > upper_bound
            // if current_waste > best_waste, then backtrack.  However, only backtrack if
            // it's high fee_rate environment.  During low fee environments, a utxo may
            // have negative waste, therefore adding more utxos in such an environment
            // may still result in reduced waste.
            || current_waste > best_waste && weighted_utxos[0].weight as f32 * options.target_feerate > long_term_feerate
        {
            dbg!("backtrack");
            backtrack = true;
        } else if weight > options.max_selection_weight {
            dbg!("weight exceeded");
            max_tx_weight_exceeded = true;
            backtrack = true;
        }
        // * value meets or exceeds the target.
        //   Record the solution and the waste then continue.
        else if value >= options.target_value {
            dbg!("value is enough, record solution");
            backtrack = true;

            // cast ok, the value and target range is (0..MAX_MONEY).
            let waste = (value as i64)
                .checked_sub(options.target_value as i64)
                .ok_or(SelectionError::AbnormallyHighAmount)?;
            current_waste = current_waste
                .checked_add(waste)
                .ok_or(SelectionError::AbnormallyHighAmount)?;

            // Check if index_selection is better than the previous known best, and
            // update best_selection accordingly.
            if current_waste <= best_waste {
                best_selection = index_selection.clone();
                best_waste = current_waste;
            }

            current_waste = current_waste
                .checked_sub(waste)
                .ok_or(SelectionError::AbnormallyHighAmount)?;
        }
        // * Backtrack
        if backtrack {
            dbg!("backtrack");
            if index_selection.is_empty() {
                return index_to_utxo_list(
                    iteration,
                    best_selection,
                    max_tx_weight_exceeded,
                    waste,
                );
            }

            loop {
                index -= 1;

                if index <= *index_selection.last().unwrap() {
                    dbg!("index under index last selection");
                    break;
                }

                let eff_value = weighted_utxos[index].effective_value(options.target_feerate)?;
                dbg!("adding effective value to available value");
                dbg!(&eff_value);
                available_value += eff_value;
            }

            assert_eq!(index, *index_selection.last().unwrap());
            let eff_value = weighted_utxos[index].effective_value(options.target_feerate)?;
            let utxo_waste = ((weighted_utxos[index].weight as f32 * options.target_feerate)
                - long_term_feerate) as i64;
            let utxo_weight = weighted_utxos[index].weight;
            current_waste = current_waste
                .checked_sub(utxo_waste)
                .ok_or(SelectionError::AbnormallyHighAmount)?;
            value = value
                .checked_sub(eff_value)
                .ok_or(SelectionError::AbnormallyHighAmount)?;
            weight -= utxo_weight;
            index_selection.pop().unwrap();
        }
        // * Add next node to the inclusion branch.
        else {
            dbg!("inclusion branch");
            let eff_value = weighted_utxos[index].effective_value(options.target_feerate)?;
            let utxo_weight = weighted_utxos[index].weight;
            let utxo_waste = ((weighted_utxos[index].weight as f32 * options.target_feerate)
                - long_term_feerate) as i64;

            // unchecked sub is used her for performance.
            // The bounds for available_value are at most the sum of utxos
            // and at least zero.
            available_value -= eff_value;

            // Check if we can omit the currently selected depending on if the last
            // was omitted.  Therefore, check if index_selection has a previous one.
            if index_selection.is_empty()
                // Check if the previous UTXO was included.
                || index - 1 == *index_selection.last().unwrap()
                // Check if the previous UTXO has the same value has the previous one.
                || weighted_utxos[index].effective_value(options.target_feerate)? != weighted_utxos[index - 1].effective_value(options.target_feerate)?
            {
                index_selection.push(index);
                current_waste = current_waste
                    .checked_add(utxo_waste)
                    .ok_or(SelectionError::AbnormallyHighAmount)?;

                // unchecked add is used here for performance.  Since the sum of all utxo values
                // did not overflow, then any positive subset of the sum will not overflow.
                value += eff_value;
                weight += utxo_weight;
            }
        }

        // no overflow is possible since the iteration count is bounded.
        index += 1;
        iteration += 1;
    }

    index_to_utxo_list(iteration, best_selection, max_tx_weight_exceeded, waste)
}

fn index_to_utxo_list(
    iterations: u32,
    selected_inputs: Vec<usize>,
    max_tx_weight_exceeded: bool,
    waste: f32,
) -> Result<SelectionOutput, SelectionError> {
    if selected_inputs.is_empty() {
        if iterations == ITERATION_LIMIT {
            Err(SelectionError::IterationLimitReached)
        } else if max_tx_weight_exceeded {
            Err(SelectionError::MaxWeightExceeded)
        } else {
            Err(SelectionError::NoSolutionFound)
        }
    } else {
        Ok(SelectionOutput {
            selected_inputs,
            waste: WasteMetric(waste),
            iterations,
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{
        algorithms::bnb::select_coin_bnb,
        types::{CoinSelectionOpt, ExcessStrategy, OutputGroup, SelectionError},
    };

    fn setup_basic_output_groups() -> Vec<OutputGroup> {
        vec![
            OutputGroup {
                value: 1000,
                weight: 100,
                input_count: 1,
                creation_sequence: None,
            },
            OutputGroup {
                value: 2000,
                weight: 200,
                input_count: 1,
                creation_sequence: None,
            },
            OutputGroup {
                value: 3000,
                weight: 300,
                input_count: 1,
                creation_sequence: None,
            },
        ]
    }

    fn bnb_setup_options(target_value: u64) -> CoinSelectionOpt {
        CoinSelectionOpt {
            target_value,
            target_feerate: 0.5, // Simplified feerate
            long_term_feerate: Some(0.5),
            min_absolute_fee: 500,
            base_weight: 10,
            change_weight: 50,
            change_cost: 10,
            avg_input_weight: 40,
            avg_output_weight: 20,
            min_change_value: 500,
            excess_strategy: ExcessStrategy::ToChange,
            max_selection_weight: 21_000_000 * 100_000_000,
            max_value: 21_000_000 * 100_000_000,
        }
    }

    fn test_bnb_solution() {
        // Define the test values
        let mut values = [
            OutputGroup::new(55000, 500),
            OutputGroup::new(400, 200),
            OutputGroup::new(40000, 300),
            OutputGroup::new(25000, 100),
            OutputGroup::new(35000, 150),
            OutputGroup::new(600, 250),
            OutputGroup::new(30000, 120),
            OutputGroup::new(94730, 50),
            OutputGroup::new(29810, 500),
            OutputGroup::new(78376, 200),
            OutputGroup::new(17218, 300),
            OutputGroup::new(13728, 100),
        ];

        // Adjust the target value to ensure it tests for multiple valid solutions
        let opt = bnb_setup_options(196282);
        let ans = select_coin_bnb(&values, &opt);
        values.sort_by_key(|v| v.value);
        let expected_solution = vec![1, 3, 4, 6, 9];
        assert_eq!(ans.unwrap().selected_inputs, expected_solution);
    }

    fn test_bnb_no_solution() {
        let inputs = setup_basic_output_groups();
        let total_input_value: u64 = inputs.iter().map(|input| input.value).sum();
        let impossible_target = total_input_value + 1000;
        let options = bnb_setup_options(impossible_target);
        let result = select_coin_bnb(&inputs, &options);
        assert!(
            matches!(result, Err(SelectionError::InsufficientFunds)),
            "Expected \"InsufficientFound\" error, got {:?}",
            result
        );
    }

    #[test]
    fn test_bnb() {
        test_bnb_solution();
        test_bnb_no_solution();
    }
}
