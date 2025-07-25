mod frb_generated; /* AUTO INJECTED BY flutter_rust_bridge. This line may not be accurate, and you can change it according to your needs. */

/// Collection of coin selection algorithms including Knapsack, Branch and Bound (BNB), First-In First-Out (FIFO), Single-Random-Draw (SRD), and Lowest Larger
pub mod algorithms;
/// Logger that allows to send the rust logs to Dart
pub mod logger;
/// Wrapper API that runs all coin selection algorithms in parallel and returns the result with lowest waste
pub mod selectcoin;
/// Core types and structs used throughout the library including OutputGroup and CoinSelectionOpt
pub mod types;
/// Helper functions with tests for fee calculation, weight computation, and waste metrics
pub mod utils;
