function KL = symmetrised_KL_multiNormal(mu0, cov0, covInv0, covDet0, mu1, cov1, covInv1, covDet1)
%% Symmetrised Kullback-Leibler divergence for multivariate normal distribution
% Arguments:
% - mu: column vector of the mean along each dimension
% - cov: square mqtrix, the covariance of the multivariate distribution
% - covInv: the inverse of the covariance matrix
% - covDet: the determinant of the covariance matrix

KL0 = KL_multiNormal(mu0, cov0, covDet0, mu1, covInv1, covDet1);
KL1 = KL_multiNormal(mu1, cov1, covDet1, mu0, covInv0, covDet0);

KL = KL0 + KL1;