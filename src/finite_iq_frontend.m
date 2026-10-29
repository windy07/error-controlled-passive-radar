function output = finite_iq_frontend(reference,surveillance,source,search,options)
sampleRate=source.sampleRate;lightSpeed=299792458;
captureLength=numel(reference);
if options.directCancelHalfWidth>0
    shifts=-options.directCancelHalfWidth:options.directCancelHalfWidth;
    candidates=zeros(captureLength,numel(shifts),'like',reference);
    for j=1:numel(shifts),candidates(:,j)=circshift(reference,shifts(j));end
    [basis,~]=qr(candidates,0);
    surveillance=surveillance-basis*(basis'*surveillance);
end
referenceTiming=template_peak(reference,source.pssWaveform,search.reference(1),search.reference(2));
monitorWindow=round(referenceTiming)+search.excess;
surveillanceTiming=template_peak(surveillance,source.pssWaveform,monitorWindow(1),monitorWindow(2));
coarseSamples=surveillanceTiming-referenceTiming;
fineSymbol=source.firstFineSymbol;
cp=source.info.CyclicPrefixLengths;
lengths=source.info.SymbolLengths;
lengths=repmat(lengths,1,ceil(size(source.fineKnown,2)/numel(lengths)));
cp=repmat(cp,1,ceil(size(source.fineKnown,2)/numel(cp)));
usefulStart=sum(lengths(1:fineSymbol-1))+cp(fineSymbol);
referenceGate=round(referenceTiming)-floor(cp(fineSymbol)/2);
surveillanceGate=round(surveillanceTiming)-floor(cp(fineSymbol)/2);
referenceRows=referenceGate+usefulStart+(1:source.info.Nfft);
surveillanceRows=surveillanceGate+usefulStart+(1:source.info.Nfft);
assert(min(referenceRows)>=1 && min(surveillanceRows)>=1 && max(surveillanceRows)<=captureLength);
referenceFrequency=fftshift(fft(reference(referenceRows)));
surveillanceFrequency=fftshift(fft(surveillance(surveillanceRows)));
rows=find(source.fineKnown(:,fineSymbol)~=0);
carrierIndex=rows-1-size(source.fineKnown,1)/2;
fftRows=carrierIndex+source.info.Nfft/2+1;
pilot=source.fineKnown(rows,fineSymbol);
integerFrequency=(carrierIndex-carrierIndex(1))/source.comb;
[fineReference,referenceProfile]=periodic_peak(referenceFrequency(fftRows).*conj(pilot),integerFrequency,options.delayFFT);
[fineSurveillance,surveillanceProfile]=periodic_peak(surveillanceFrequency(fftRows).*conj(pilot),integerFrequency,options.delayFFT);
fineDelay=mod(lightSpeed*(surveillanceGate-referenceGate)/sampleRate+source.period*(fineSurveillance-fineReference),source.period);
allReference=[];allSurveillance=[];allCarriers=[];
for symbol=find(any(source.fineKnown~=0,1))
    start=sum(lengths(1:symbol-1))+cp(symbol);
    rIndex=referenceGate+start+(1:source.info.Nfft);
    sIndex=surveillanceGate+start+(1:source.info.Nfft);
    rFrequency=fftshift(fft(reference(rIndex)));
    sFrequency=fftshift(fft(surveillance(sIndex)));
    selected=find(source.fineKnown(:,symbol)~=0);
    k=selected-1-size(source.fineKnown,1)/2;
    selectedFFT=k+source.info.Nfft/2+1;
    coefficients=conj(source.fineKnown(selected,symbol));
    allReference=[allReference;rFrequency(selectedFFT).*coefficients];
    allSurveillance=[allSurveillance;sFrequency(selectedFFT).*coefficients];
    allCarriers=[allCarriers;k];
end
fullComb=0;
for j=2:numel(allCarriers),fullComb=gcd(fullComb,allCarriers(j)-allCarriers(1));end
integerFrequency=(allCarriers-allCarriers(1))/fullComb;
[fullReference,fullReferenceProfile]=periodic_peak(allReference,integerFrequency,options.delayFFT);
[fullSurveillance,fullSurveillanceProfile]=periodic_peak(allSurveillance,integerFrequency,options.delayFFT);
fullPeriod=lightSpeed/(30000*fullComb);
fullDelay=mod(lightSpeed*(surveillanceGate-referenceGate)/sampleRate+fullPeriod*(fullSurveillance-fullReference),fullPeriod);
correlation=ifft(fft(surveillance).*conj(fft(reference)));
correlation=interpft(correlation,options.cafOversample*captureLength);
lags=options.cafOversample*search.excess(1):options.cafOversample*search.excess(2);
indices=mod(lags,numel(correlation))+1;
[~,best]=max(abs(correlation(indices)).^2);
cafSamples=lags(best);
if best>1 && best<numel(indices)
    cafSamples=cafSamples+parabolic_offset(abs(correlation(indices(best+(-1:1)))).^2);
end
cafSamples=cafSamples/options.cafOversample;
output=struct('coarse',lightSpeed*coarseSamples/sampleRate,'fine',fineDelay, ...
    'fullFine',fullDelay,'fullFinePeriod',fullPeriod, ...
    'caf',lightSpeed*cafSamples/sampleRate,'referenceTiming',referenceTiming,'surveillanceTiming',surveillanceTiming, ...
    'referenceGate',referenceGate,'surveillanceGate',surveillanceGate,'fineSymbol',fineSymbol, ...
    'fineReferenceFraction',fineReference,'fineSurveillanceFraction',fineSurveillance, ...
    'fullReferenceFraction',fullReference,'fullSurveillanceFraction',fullSurveillance);
if options.saveProfiles
    output.referenceProfile=referenceProfile;output.surveillanceProfile=surveillanceProfile;
    output.fullReferenceProfile=fullReferenceProfile;output.fullSurveillanceProfile=fullSurveillanceProfile;
    output.cafProfile=abs(correlation(indices)).^2;
    output.cafLags=lags/options.cafOversample;
end
end

function location=template_peak(record,template,lower,upper)
count=2^nextpow2(numel(record)+numel(template)-1);
correlation=ifft(fft(record,count).*conj(fft(template,count)));
indices=mod(lower:upper,count)+1;
[~,best]=max(abs(correlation(indices)).^2);
location=lower+best-1;
if best>1 && best<numel(indices)
    location=location+parabolic_offset(abs(correlation(indices(best+(-1:1)))).^2);
end
end

function [fraction,profile]=periodic_peak(values,integerFrequency,count)
frequencyGrid=accumarray(mod(integerFrequency,count)+1,values,[count,1],@sum,0);
profile=abs(ifft(frequencyGrid)*count).^2;
[~,best]=max(profile);
neighbors=mod(best-1+(-1:1),count)+1;
fraction=mod(best-1+parabolic_offset(profile(neighbors)),count)/count;
end

function value=parabolic_offset(values)
value=.5*(values(1)-values(3))/(values(1)-2*values(2)+values(3));
end
