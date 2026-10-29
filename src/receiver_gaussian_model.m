function model = receiver_gaussian_model(basis, calibrationMean, calibrationCovariance, noiseCovariance, periods)
sources = size(basis,1);
assert(size(noiseCovariance,1) == 2*sources && size(noiseCovariance,2) == 2*sources);
assert(numel(periods) == sources && all(periods > 0));
noiseRoot = chol(noiseCovariance,'lower');
jointBasis = [basis;basis];
covariance = noiseCovariance + jointBasis*calibrationCovariance*jointBasis';
coarseRoot = chol(covariance(1:sources,1:sources),'lower');
conditionalGain = (coarseRoot' \ (coarseRoot \ covariance(1:sources,sources+1:end)))';
conditionalCovariance = covariance(sources+1:end,sources+1:end) ...
    - conditionalGain*covariance(1:sources,sources+1:end);
conditionalCovariance = (conditionalCovariance+conditionalCovariance')/2;
chol(conditionalCovariance);
whitenedBasis = noiseRoot \ jointBasis;
[Q,~] = qr(whitenedBasis);
rankBasis = size(basis,2);
assert(rank(whitenedBasis) == rankBasis,'Calibration basis must have independent columns.');
contrast = Q(:,rankBasis+1:end)' / noiseRoot;
periodInjection = [zeros(sources);diag(periods)];
contrastInjection = contrast*periodInjection;
integerPrecision = contrastInjection'*contrastInjection;
integerRoot = chol(integerPrecision,'lower');
noiseCoarseRoot = chol(noiseCovariance(1:sources,1:sources),'lower');
noiseGain = (noiseCoarseRoot' \ (noiseCoarseRoot \ noiseCovariance(1:sources,sources+1:end)))';
weightedBasis = noiseCoarseRoot \ basis;
calibrationRoot = chol(weightedBasis'*weightedBasis,'lower');
flatCalibrationCovariance = calibrationRoot' \ (calibrationRoot \ eye(rankBasis));
flatCalibrationGain = flatCalibrationCovariance * (weightedBasis'/noiseCoarseRoot);
adjustedBasis = (eye(sources)-noiseGain)*basis;
flatConditionalGain = noiseGain+adjustedBasis*flatCalibrationGain;
flatConditionalCovariance = noiseCovariance(sources+1:end,sources+1:end) ...
    - noiseGain*noiseCovariance(1:sources,sources+1:end) ...
    + adjustedBasis*flatCalibrationCovariance*adjustedBasis';
flatConditionalCovariance = (flatConditionalCovariance+flatConditionalCovariance')/2;
chol(flatConditionalCovariance);
model = struct('basis',basis,'calibrationMean',calibrationMean(:),'calibrationCovariance',calibrationCovariance, ...
    'noiseCovariance',noiseCovariance,'covariance',covariance,'meanBias',basis*calibrationMean(:), ...
    'coarseRoot',coarseRoot,'conditionalGain',conditionalGain,'conditionalCovariance',conditionalCovariance, ...
    'periods',periods(:),'contrast',contrast,'contrastInjection',contrastInjection, ...
    'integerCovariance',integerRoot' \ (integerRoot \ eye(sources)), ...
    'noiseCoarseRoot',noiseCoarseRoot,'flatCalibrationGain',flatCalibrationGain, ...
    'flatConditionalGain',flatConditionalGain,'flatConditionalCovariance',flatConditionalCovariance);
end
