function result = demo_localization(outputDirectory)
root = fileparts(mfilename('fullpath'));
if nargin == 0
    outputDirectory = fullfile(root,'outputs','localization');
end
assert(~isfolder(outputDirectory) && ~isfile(outputDirectory));
originalPath = path;
restorePath = onCleanup(@() path(originalPath));
addpath(fullfile(root,'src'),'-begin');
data = load(fullfile(root,'data','example_input.mat'));
options = struct('prune',true,'factorTolerance',1e-10,'posteriorTolerance',1e-5, ...
    'batchSize',128,'maxExpansions',20000,'envelope',"prefix",'backend',"sphere_batch", ...
    'boxRadius',3,'progressSeconds',60,'factorOrder',"variance",'pairEnvelope',true);
estimate = spatial_grid_posterior_joint(data.coarse,data.fine,data.roi, ...
    data.spacing,data.transmitters,data.receivers,data.models,"marginal", ...
    options,data.leafPoints,data.maximumBlocks,data.initialPosition);
reference = load(fullfile(root,'data','example_reference.mat'));
diameter = norm(data.roi(:,2)-data.roi(:,1));
comparison = struct('meanDifference',norm(estimate.mean-reference.mean), ...
    'sumAnalyticMeanAllowances',diameter*(estimate.tvBound+reference.tvBound), ...
    'positionError',norm(estimate.mean-reference.position));
result = struct('estimate',estimate,'comparison',comparison,'options',options);
assert(estimate.converged);
assert(comparison.meanDifference <= comparison.sumAnalyticMeanAllowances);
mkdir(outputDirectory);
save(fullfile(outputDirectory,'localization.mat'),'result');
fprintf('TV bound %.9g, reference difference %.9g m, position error %.9g m.\n', ...
    estimate.tvBound,comparison.meanDifference,comparison.positionError);
end
