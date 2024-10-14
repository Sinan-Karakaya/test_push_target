KMC tries to computes K clusters for our data.
K is predefined by us.
Let's see how the computer computes the clusters.

1. Choose K random spots to be the ***centroids*** (center of the clusters)
2. Calculate the distance from each points to the centroids, and assign those points to the closest centroid, this create a cluster
3. Compute new centroids based on the average position of each points in each cluster
4. Repeat step 2 until the centroids barely move

This process is known as ***Expectation Maximization***. The step 2 of the process above is the *Expectation*, and the step 3 is our *Maximization* step.

This is our first example of [[Unsupervised learning]].

## Why ?

This would then let our model make guesses on the category of a given new data based on the position relative to the clusters.

![[Pasted image 20231122160525.png]]

