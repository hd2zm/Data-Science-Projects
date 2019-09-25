# Clustering Amazon Book Reviews from 2005

This project filters out book reviews by 
* year (2005)
* rating (3 stars are excluded)
* helpfulness (least helpful are excluded)

After getting 15,000 reviews, non-negative matrix factorization was applied to generate 30 different topics. These topics were grouped into two categories: content-based reviews and assessment-based reviews. Content-based reviews gave descriptions of the book while assessment-based reviews gave descriptions of how the reviewer felt about the book. 

PCA was applied to reduce the dimensions from 30 to 2, and K-means was subsequently applied to find any meaningful clusters from that information. 

The goal of this project was to see if it is possible to visualize different groupings of the reviews. Unfortunately, no significant insights were found. 
