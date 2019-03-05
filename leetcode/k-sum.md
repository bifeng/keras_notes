https://cs.stackexchange.com/questions/2973/generalised-3sum-k-sum-problem

k-SUM can be solved more quickly as follows:  
For even k: Compute a sorted list S of all sums of k/2 input elements. 
Check whether S contains both some number x and its negation −x. 
The algorithm runs in $O(n^{k/2} \log n)​$ time.

For odd k: Compute the sorted list S of all sums of (k−1)/2 input elements. 
For each input element a, check whether S contains both x and −a−x, for some number x. 
(The second step is essentially the $O(n^2)$-time algorithm for 3SUM.) 
The algorithm runs in $O(n^{(k+1)}/2)​$ time.

Both algorithms are optimal (except possibly for the log factor when k is even and bigger than 2) for any constant k in a certain weak but natural restriction of the linear decision tree model of computation. For more details, see:

Nir Ailon and Bernard Chazelle. Lower bounds for linear degeneracy testing. JACM 2005.

Jeff Erickson. Lower bounds for linear satisfiability problems. CJTCS 1999.
