function [logUpper,logInflation] = wrapped_gaussian_majorant(residual,covariance,periods,method)
periods = periods(:);
dimension = numel(periods);
normalized = covariance./(periods*periods');
root = chol(normalized,'lower');
diagonal = sum(abs(normalized),2);
logInflation = .5*sum(log(diagonal))-sum(log(diag(root)));
if method=="maximum"
    wrapped = mod(residual./periods+.5,1)-.5;
    logUpper = -sum(log(diag(root)))-dimension/2*log(2*pi)-sum(log(periods)) ...
        -.5*sum(wrapped.^2./diagonal,1);
else
    logUpper = (logInflation-sum(log(periods)))*ones(1,size(residual,2));
    for source = 1:dimension
        [lower,error] = wrapped_normal_interval(residual(source,:)/periods(source), ...
            diagonal(source),1,1e-10);
        center = max(lower,error);
        logUpper = logUpper+center+log(exp(lower-center)+exp(error-center));
    end
end
end
