function transform=integer_lattice_reduction(basis)
dimension=size(basis,2);
transform=eye(dimension);
index=2;iterations=0;
while index<=dimension
    iterations=iterations+1;
    assert(iterations<=20000,'Lattice:LLLLimit','LLL iteration limit exceeded.');
    for previous=index-1:-1:1
        [~,upper]=qr(basis*transform,0);
        multiple=round(upper(previous,index)/upper(previous,previous));
        transform(:,index)=transform(:,index)-multiple*transform(:,previous);
    end
    [~,upper]=qr(basis*transform,0);
    mu=upper(index-1,index)/upper(index-1,index-1);
    if upper(index,index)^2>=(.75-mu^2)*upper(index-1,index-1)^2*(1-32*eps)
        index=index+1;
    else
        transform(:,[index-1,index])=transform(:,[index,index-1]);
        index=max(2,index-1);
    end
end
assert(max(abs(transform),[],'all')<2^45,'Lattice:IntegerRange', ...
    'Integer transform exceeds the checked exact range.');
end
