function KL = KL_multiNormal(mu0, cov0, covDet0, mu1, covInv1, covDet1)
%% Kulback-Leibler divergence for multivariate normal distributions
% Arguments:
% - mu: column vector of the mean along each dimension
% - cov: square mqtrix, the covariance of the multivariate distribution
% - covInv: the inverse of the covariance matrix
% - covDet: the determinant of the covariance matrix

% Check that mu0 and mu1 are column vectors
assert(size(mu0, 2) == 1 && size(mu1, 2) == 1)

k = size(cov0, 1); % Dimension of the normal distributions

KL = 1/2 * (trace(covInv1 * cov0) + (mu1 - mu0)' * covInv1 * (mu1 - mu0) - k + log(covDet1 / covDet0));