classdef FrozenNet < dagnn.ElementWise
  properties
	pathmodel = '';
	output_point = '';
	model = [];
  end

  methods
    function outputs = forward(obj, inputs, params)
        obj.model.mode = 'test' ;
        obj.model.conserveMemory = true;
        obj.model.vars(obj.model.getVarIndex(obj.output_point)).precious = true;
        
    	obj.model.eval({obj.model.vars(1).name, inputs{1}});
        outputs{1} = obj.model.vars(obj.model.getVarIndex(obj.output_point)).value;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        derInputs = {[]} ;
        derParams = {} ;

    end

    % ---------------------------------------------------------------------
    function obj = FrozenNet(varargin)
      obj.load(varargin{:}) ;
      
      if(~isempty(varargin))
          m = load(obj.pathmodel);
          fields = fieldnames(m);
          obj.model = dagnn.DagNN.loadobj(m.(fields{1})) ;
          %obj.model.removeLayer(obj.model.layers(end).name);
          obj.model.move('gpu');
        
      end
    end

    function obj = reset(obj)
      %reset@dagnn.ElementWise(obj) ;
       obj.model.reset();
    end
  end
end
