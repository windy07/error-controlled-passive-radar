function source = nr_opportunity_burst(kind,seed,slots)
stream=RandStream('mt19937ar','Seed',seed);
carrier=nrCarrierConfig('NSizeGrid',52,'SubcarrierSpacing',30,'NCellID',31+kind);
subcarriers=carrier.NSizeGrid*12;
grid=complex(zeros(subcarriers,14*slots));known=zeros(size(grid));
pssKnown=zeros(size(grid));fineKnown=zeros(size(grid));
for slot=0:slots-1
    carrier.NSlot=slot;
    bits=randi(stream,[0,1],2*subcarriers*14,1);
    data=((1-2*bits(1:2:end))+1i*(1-2*bits(2:2:end)))/sqrt(2);
    localGrid=reshape(data,subcarriers,14);
    pilots=zeros(size(localGrid));
    if kind<=3
        csirs=nrCSIRSConfig('CSIRSType','nzp','CSIRSPeriod','on','NumRB',52, ...
            'RBOffset',0,'SymbolLocations',13,'SubcarrierLocations',0,'NID',carrier.NCellID);
        if kind==1
            csirs.RowNumber=2;csirs.Density='one';
        elseif kind==2
            csirs.RowNumber=1;csirs.Density='three';
        else
            csirs.RowNumber=2;csirs.Density='dot5even';
        end
        indices=nrCSIRSIndices(carrier,csirs);values=nrCSIRS(carrier,csirs);
    else
        prs=nrPRSConfig('PRSResourceSetPeriod','on','NumRB',52,'RBOffset',0, ...
            'CombSize',6,'NumPRSSymbols',6,'SymbolStart',6,'REOffset',0,'NPRSID',carrier.NCellID);
        indices=nrPRSIndices(carrier,prs);values=nrPRS(carrier,prs);
    end
    localGrid(indices)=values;pilots(indices)=values;
    columns=slot*14+(1:14);
    grid(:,columns)=localGrid;fineKnown(:,columns)=pilots;
end
pssRows=(subcarriers-240)/2+(57:183);
pssKnown(pssRows,3)=nrPSS(carrier.NCellID);
grid(pssRows,3)=pssKnown(pssRows,3);
known=pssKnown+fineKnown;
carrier.NSlot=0;
[waveform,info]=nrOFDMModulate(carrier,grid,'Nfft',1024,'Windowing',0);
pssWaveform=nrOFDMModulate(carrier,pssKnown,'Nfft',1024,'Windowing',0);
fineWaveform=nrOFDMModulate(carrier,fineKnown,'Nfft',1024,'Windowing',0);
firstFineSymbol=find(any(fineKnown~=0,1),1);
rows=find(fineKnown(:,firstFineSymbol)~=0);
indices=rows-1-subcarriers/2;
comb=0;
for j=2:numel(indices),comb=gcd(comb,indices(j)-indices(1));end
assert(comb>0);
source=struct('kind',kind,'seed',seed,'slots',slots,'carrier',carrier,'grid',grid,'known',known, ...
    'pssKnown',pssKnown,'fineKnown',fineKnown,'waveform',waveform,'pssWaveform',pssWaveform, ...
    'fineWaveform',fineWaveform,'info',info,'firstFineSymbol',firstFineSymbol, ...
    'comb',comb,'period',299792458/(comb*30000),'sampleRate',info.SampleRate, ...
    'centerFrequency',3.30e9+(kind-1)*60e6);
source.receiver=struct('sampleRate',info.SampleRate,'info',info,'pssWaveform',pssWaveform, ...
    'fineKnown',fineKnown,'firstFineSymbol',firstFineSymbol,'comb',comb,'period',source.period);
end
