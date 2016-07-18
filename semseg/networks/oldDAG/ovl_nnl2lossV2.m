function Y = vl_nnl2lossV2(X,c, dzdy)
	% convert from standard form to heat form

	if nargin <= 2
        %size(W)
        %size(X)
        %size(newLabs)
		a = abs(X-c);
        %a = (X-newLabs);
		%a=a.^2;
		Y=sum(a(:)) ./ numel(X);
		%Y=Y/size(X,4);

    else
        Y = dzdy .* sign(X-c) ./ numel(X);
	%	Y = (W.*(X-newLabs).*(W)) .* dzdy;
        %Y = (1/N)*(X-newLabs) * dzdy;
	end
end
