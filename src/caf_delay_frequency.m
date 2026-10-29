function output=caf_delay_frequency(reference,surveillance,sampleRate,delayLimits,frequencyLimits)
clock=tic;count=numel(reference);time=(0:count-1)'/sampleRate;
assert(numel(surveillance)==count && all(delayLimits==round(delayLimits)) && diff(frequencyLimits)>0);
referenceSpectrum=conj(fft(reference));
lags=delayLimits(1):delayLimits(2);indices=mod(lags,count)+1;
coarseFrequencies=frequencyLimits(1):1000:frequencyLimits(2);
assert(numel(coarseFrequencies)>=3 && coarseFrequencies(end)==frequencyLimits(2));
coarsePowers=zeros(size(coarseFrequencies));evaluations=0;
for index=1:numel(coarseFrequencies)
    correlation=correlate(coarseFrequencies(index));
    coarsePowers(index)=max(abs(correlation(indices)).^2);
end
[~,best]=max(coarsePowers);
fineFrequencies=max(frequencyLimits(1),coarseFrequencies(best)-1000):250: ...
    min(frequencyLimits(2),coarseFrequencies(best)+1000);
finePowers=zeros(size(fineFrequencies));
for index=1:numel(fineFrequencies)
    existing=find(coarseFrequencies==fineFrequencies(index),1);
    if isempty(existing)
        correlation=correlate(fineFrequencies(index));
        finePowers(index)=max(abs(correlation(indices)).^2);
    else
        finePowers(index)=coarsePowers(existing);
    end
end
[~,best]=max(finePowers);frequency=fineFrequencies(best);
if best>1 && best<numel(fineFrequencies)
    three=finePowers(best+(-1:1));
    frequency=frequency+250*.5*(three(1)-three(3))/(three(1)-2*three(2)+three(3));
end
correlation=correlate(frequency);correlation=interpft(correlation,8*count);
fineLags=8*delayLimits(1):8*delayLimits(2);fineIndices=mod(fineLags,numel(correlation))+1;
[~,best]=max(abs(correlation(fineIndices)).^2);delay=fineLags(best);
if best>1 && best<numel(fineIndices)
    three=abs(correlation(fineIndices(best+(-1:1)))).^2;
    delay=delay+.5*(three(1)-three(3))/(three(1)-2*three(2)+three(3));
end
output=struct('range',299792458*delay/(8*sampleRate),'frequency',frequency, ...
    'seconds',toc(clock),'frequencyEvaluations',evaluations, ...
    'frequencyAtBoundary',frequency==frequencyLimits(1)||frequency==frequencyLimits(2), ...
    'delayAtBoundary',best==1||best==numel(fineIndices), ...
    'coarseFrequencies',coarseFrequencies,'coarsePowers',coarsePowers, ...
    'fineFrequencies',fineFrequencies,'finePowers',finePowers);

    function correlation=correlate(frequency)
        correlation=ifft(fft(surveillance.*exp(-2i*pi*frequency*time)).*referenceSpectrum);
        evaluations=evaluations+1;
    end
end
