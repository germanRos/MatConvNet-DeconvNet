function y = vl_nnLp(x, c, p, dzdy)

	disp('--- loss ----');
	size(x)
	size(c)
	disp('---end loss ---');
	c = reshape(c, size(x));
	d = bsxfun(@minus, x, c);

	% forward mode
	if isempty(dzdy)
        d = d(:);
        
		% L1
		if p == 1
			y = sum(abs(d));
		elseif p == 2
			y = sum(d.*d);
		else
			y = sum(abs(d).^p) ;
		end
	% backward mode
	else
		if p == 1
			y = bsxfun(@times, dzdy, sign(d)) ;
		elseif p == 2
			y = bsxfun(@times, 2 * dzdy, d) ;
		elseif p < 1
			y = bsxfun(@times, p * dzdy, max(abs(d), opts.epsilon).^(p-1) .* sign(d)) ;
		else
			y = bsxfun(@times, p * dzdy, abs(d).^(p-1) .* sign(d)) ;
		end
	end
end
