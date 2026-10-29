function prediction=passive_geometry_grid(points,transmitters,receivers)
M=size(transmitters,1); N=size(receivers,1); P=size(points,2);
sourceDistance=zeros(M,P); receiverDistance=zeros(N,P);
for m=1:M, sourceDistance(m,:)=vecnorm(points-transmitters(m,:)',2,1); end
for n=1:N, receiverDistance(n,:)=vecnorm(points-receivers(n,:)',2,1); end
prediction=zeros(M*N,P);
for n=1:N
    prediction((n-1)*M+(1:M),:)=sourceDistance+receiverDistance(n,:)-vecnorm(transmitters-receivers(n,:),2,2);
end
end
