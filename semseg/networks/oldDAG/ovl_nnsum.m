function [Y] = vl_nnsum(inputs, dzdy)
	% forward mode
	
	if nargin < 2
		Y = gpuArray.zeros(size(inputs{1}), 'single');
		for k = 1:numel(inputs)
            	%size(inputs{k})
        	Y = Y + inputs{k};
      	end
	% backward mode
	else
		Y = cell(numel(inputs), 1);
		for k = 1:numel(inputs)
			Y{k} = dzdy;
		end
	end

end
