function K = multinomial_kernel(data1, data2, order, beta, ganma)
%% Multinomial Kernel
% Arguments:
% - data1: data matrix where rows are observations and columns are features
% - data2: same as data1
% - order: the order of the multinomial expansion
% - beta and ganma: no idea how to call them... coefficients? Default value
% removes multiple instances of the same polynome


assert(size(data1, 2) == size(data2, 2))

if nargin<4
    if order ~= 1
        beta = order.^(1/(order-1));
        ganma = order.^(order/(1-order));
    else
        beta = 1;
        ganma = 1;
    end
end

K = ganma*( ((1 + beta .* (data2 * data1')) .^ order) - 1);
