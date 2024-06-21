Simulations to accompany the paper "Auction Timing and Market Thickness" (Games and Economic Behavior, 2024, Chaves and Ichihashi).

In that paper we prove that the elasticity of a value distribution determines whether profit-maximizing auctioneers choose inefficiently high or inefficiently low latencies.
The result is derived for single-unit supply and single-unit-demand bidders, and the proof is hard to adapt to multi-unit auctions.

For generalized Pareto-distributed-value single-unit-demand bidders, and k+1-price auctions of k identical items, these Python simulations show that the Revenue from profit-maximizing auctions has a slower (faster) growth rate than the social surplues from an efficient auction when the elasticity of the value distribution is increasing (decreasing).

To explore the parameter space efficiently and with minimal error, the various functions (densities, order statistics, etc) are compiled to C via Numba, operations are parallelized via the multiprocessing library, and moments of order statistics are computed via quadrature.
