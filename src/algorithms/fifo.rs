use crate::{
    types::{CoinSelectionOpt, OutputGroup, SelectionError, SelectionOutput, WasteMetric},
    utils::{calculate_fee, calculate_waste, sum},
};

/// Performs coin selection using the First-In-First-Out (FIFO) algorithm.
/// The selection can produce a change output
/// Returns `NoSolutionFound` if no solution is found.
pub fn select_coin_fifo(
    inputs: &[OutputGroup],
    options: &CoinSelectionOpt,
) -> Result<SelectionOutput, SelectionError> {
    let mut accumulated_value: u64 = 0;
    let mut accumulated_weight: u64 = 0;
    let mut selected_inputs: Vec<usize> = Vec::new();
    let mut estimated_fees: u64 = 0;
    let base_fees = calculate_fee(options.base_weight, options.target_feerate).unwrap_or_default();
    let target = sum(
        options.target_value,
        base_fees.max(options.min_absolute_fee),
    )?;

    // Sorting the inputs vector based on creation_sequence
    let mut sorted_inputs: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter(|(_, og)| og.creation_sequence.is_some())
        .collect();

    sorted_inputs.sort_by(|a, b| a.1.creation_sequence.cmp(&b.1.creation_sequence));

    let inputs_without_sequence: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter(|(_, og)| og.creation_sequence.is_none())
        .collect();

    sorted_inputs.extend(inputs_without_sequence);

    let mut change_used = false;
    for (index, inputs) in sorted_inputs {
        estimated_fees = calculate_fee(accumulated_weight, options.target_feerate)?;
        let target_and_fees_without_change = sum(target, estimated_fees)?;
        if accumulated_value >= target_and_fees_without_change {
            if options
                .change_left_from_effective_value(accumulated_value - estimated_fees)
                .is_ok_and(|x| x.is_some())
            {
                // If the value is equal or above the target and the added cost of a change +
                // the dust limit, we add the weight of the change to the weight, because a change will be present
                accumulated_weight = sum(accumulated_weight, options.change_weight)?;
                // The estimated fee will be increased because of this new output.
                // Theses fees are already taken into account in the previous target
                // because the minimum target with a change include the change_cost
                estimated_fees = sum(estimated_fees, options.avg_output_weight)?;
                change_used = true;
            }
            break;
        }
        accumulated_value = sum(accumulated_value, inputs.value)?;
        accumulated_weight = sum(accumulated_weight, inputs.weight)?;
        selected_inputs.push(index);
    }
    if accumulated_value < sum(target, estimated_fees)? {
        // If the change_cost value given is under the real change cost value,
        // it will cause the accumulated_value to be lower than the required fees
        if change_used {
            Err(SelectionError::LowerThanFeeChangeCost)
        } else {
            Err(SelectionError::InsufficientFunds)
        }
    } else {
        let waste: f32 = calculate_waste(
            options,
            accumulated_value,
            accumulated_weight,
            estimated_fees,
        )?;
        Ok(SelectionOutput {
            selected_inputs,
            waste: WasteMetric(waste),
            iterations: 1,
        })
    }
}

#[cfg(test)]
mod test {

    use crate::{
        algorithms::fifo::select_coin_fifo,
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
    fn setup_output_groups_withsequence() -> Vec<OutputGroup> {
        vec![
            OutputGroup {
                value: 1000,
                weight: 100,
                input_count: 1,
                creation_sequence: Some(1),
            },
            OutputGroup {
                value: 2000,
                weight: 200,
                input_count: 1,
                creation_sequence: Some(5000),
            },
            OutputGroup {
                value: 3000,
                weight: 300,
                input_count: 1,
                creation_sequence: Some(1001),
            },
            OutputGroup {
                value: 1500,
                weight: 150,
                input_count: 1,
                creation_sequence: None,
            },
        ]
    }

    fn setup_options(target_value: u64) -> CoinSelectionOpt {
        CoinSelectionOpt {
            target_value,
            target_feerate: 0.4, // Simplified feerate
            long_term_feerate: Some(0.4),
            min_absolute_fee: 0,
            base_weight: 10,
            change_weight: 50,
            change_cost: 10,
            avg_input_weight: 20,
            avg_output_weight: 10,
            min_change_value: 500,
            excess_strategy: ExcessStrategy::ToChange,
            max_selection_weight: 10_000,
        }
    }

    fn test_successful_selection() {
        let mut inputs = setup_basic_output_groups();
        let mut options = setup_options(2500);
        let mut result = select_coin_fifo(&inputs, &options);
        assert!(result.is_ok());
        let mut selection_output = result.unwrap();
        assert!(!selection_output.selected_inputs.is_empty());

        inputs = setup_output_groups_withsequence();
        options = setup_options(500);
        result = select_coin_fifo(&inputs, &options);
        assert!(result.is_ok());
        selection_output = result.unwrap();
        assert!(!selection_output.selected_inputs.is_empty());
    }

    fn test_insufficient_funds() {
        let inputs = setup_basic_output_groups();
        let options = setup_options(7000); // Set a target value higher than the sum of all inputs
        let result = select_coin_fifo(&inputs, &options);
        assert!(matches!(result, Err(SelectionError::InsufficientFunds)));
    }

    #[test]
    fn test_fifo() {
        test_successful_selection();
        test_insufficient_funds();
    }
}
