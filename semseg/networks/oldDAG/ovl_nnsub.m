function [Y] = vl_nnsub(inputs, dzdy)
	% forward mode
	
	if nargin < 2
		Y = inputs{1} - inputs{2};
	% backward mode
	else
		Y = cell(numel(inputs), 1);
		Y{1} = dzdy;
		Y{2} = -dzdy;
	end

end
