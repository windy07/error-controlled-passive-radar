function [value, detail] = receiver_loglikelihood(coarse, fine, predicted, model, method, tolerance, maxExpansions)
sources = numel(coarse);
if method == "marginal" || method == "maximum"
    coarseResidual = coarse-predicted-model.meanBias;
    normalized = model.coarseRoot \ coarseResidual;
    coarseLog = -.5*sum(normalized.^2,1)-sum(log(diag(model.coarseRoot)))-sources/2*log(2*pi);
    fineResidual = fine-predicted-model.meanBias-model.conditionalGain*coarseResidual;
    [fineLog,detail] = wrapped_gaussian_tree(fineResidual,model.conditionalCovariance,model.periods,tolerance,maxExpansions);
    if method == "maximum"
        fineLog = detail.logMaximumComponent;
        detail.relativeErrorBound = exp(detail.logErrorBound-fineLog);
        detail.converged = detail.relativeErrorBound <= tolerance;
    end
    value = coarseLog+fineLog;
elseif method == "difference"
    residual = [coarse-predicted;fine-predicted];
    projected = model.contrast*residual;
    integerMean = model.integerCovariance*(model.contrastInjection'*projected);
    perpendicular = projected-model.contrastInjection*integerMean;
    [factor,detail] = wrapped_gaussian_tree(integerMean,model.integerCovariance,ones(sources,1),tolerance,maxExpansions);
    value = -.5*sum(perpendicular.^2,1)+factor;
else
    error('ReceiverLikelihood:Method','Unknown estimation objective.');
end
end
