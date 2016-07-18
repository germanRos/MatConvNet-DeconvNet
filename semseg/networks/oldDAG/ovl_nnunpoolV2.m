function [Y] = vl_nnunpoolV2(X, Ind, sizeY, varargin)
% This version of the unpooling allows for unpooling data with different channels
%
%
	% forward mode
	if(isempty(varargin))
		Y = gpuArray.zeros(sizeY(1), sizeY(2), size(X, 3), size(X, 4), 'single');
		disp('--- size(Y)---');
		size(Y)

		[I1, I2, I3, I4] = ind2sub(sizeY, Ind+1);
		I1r = reshape(I1, [size(X, 1), size(X, 2), sizeY(3), size(X, 4)]);
		I2r = reshape(I2, [size(X, 1), size(X, 2), sizeY(3), size(X, 4)]);
		I3r = reshape(I3, [size(X, 1), size(X, 2), sizeY(3), size(X, 4)]);
		I4r = reshape(I4, [size(X, 1), size(X, 2), sizeY(3), size(X, 4)]);

		I1r = I1r(:, :, 1:size(X, 3), :);
		I2r = I2r(:, :, 1:size(X, 3), :);
		I3r = I3r(:, :, 1:size(X, 3), :);
		I4r = I4r(:, :, 1:size(X, 3), :);
		indOUT = sub2ind(size(Y), I1r(:), I2r(:), I3r(:), I4r(:));

		size(indOUT)

		Y(indOUT) = X;

		%IndR = reshape(Ind+1, [size(X, 1), size(X, 2), sizeY(3), size(X, 4)]);
		%IndR = IndR(:,:,1:size(X, 3), :);
		%size(IndR)
		%Y(IndR) = X;
		%Y(Ind+1) = X;
		
	% backward mode
	else
		DZDX = (varargin{1});

		size(DZDX)
		max(Ind+1)
		size(Ind)
		sizeY

		Y = reshape(DZDX(Ind+1), sizeY);
	end
end
