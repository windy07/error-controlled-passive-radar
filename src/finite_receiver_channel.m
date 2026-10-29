function signal = finite_receiver_channel(waveform,sampleRate,captureLength,delaySamples,centerFrequency,calibration,cfo)
assert(captureLength>=numel(waveform)+ceil(abs(delaySamples))+128);
frequency=(0:captureLength-1)'*sampleRate/captureLength;
frequency(frequency>=sampleRate/2)=frequency(frequency>=sampleRate/2)-sampleRate;
u=centerFrequency-3.39e9+frequency;
phase=-2*pi/299792458*(calibration(1)*u+calibration(2)*u.^2/(2*90e6)+calibration(3)*u.^3/(3*90e6^2));
response=exp(1i*phase-2i*pi*frequency*delaySamples/sampleRate);
signal=ifft(fft(waveform,captureLength).*response);
signal=signal.*exp(2i*pi*cfo*(0:captureLength-1)'/sampleRate);
end
