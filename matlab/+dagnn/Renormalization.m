classdef Renormalization < dagnn.ElementWise
  properties

  end

  methods
    function outputs = forward(obj, inputs, params)
      X = inputs{1};
      outputs{1} = bsxfun(@rdivide, X, sum(X, 3)+1e-5);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
 	X = inputs{1};

	Y1 = bsxfun(@plus, -X, sum(X, 3));
	Y2 = sum(X, 3).*sum(X, 3);
	Y = bsxfun(@rdivide, Y1, Y2+1e-5);
        derInputs{1} = Y .* derOutputs{1};
      derParams = {} ;
    end

 
    function obj = Renormalization(varargin)
      obj.load(varargin) ;
    end
  end
end
