function [netout] = convertModel(model, outputfile)
	run(fullfile(fileparts(mfilename('fullpath')),'../../matlab/', 'vl_setupnn.m')) ;

    L = numel(model.layers);
    
    newmodel = dagnn.DagNN() ;
    cell_params = cell(L, 0);
    model.layers{1}.inputs = 'input';
    for l=1:L
      layer = model.layers{l};
      if(isfield(layer, 'weights'))
          for j=1:numel(layer.weights)
            cell_params{l, j} = sprintf('%sw%02d', layer.name, j); 
          end
      else
          cell_params{l, 1} = '';
      end
      
      switch(layer.type)
          case 'conv'
            newmodel.addLayer(layer.name, dagnn.Conv('size', size(layer.weights{1}), 'hasBias', (numel(layer.weights) > 1), 'stride', layer.stride, 'pad', layer.pad), layer.inputs, {layer.name},  cell_params(l, :));
          case 'convt'
            newmodel.addLayer(layer.name, dagnn.ConvTranspose('size', size(layer.weights{1}), 'hasBias', (numel(layer.weights) > 1), 'upsample', [layer.Upsample, layer.Upsample], 'crop', layer.crop), layer.inputs, {layer.name},  cell_params(l, :));         
          case 'relu'
            newmodel.addLayer(layer.name, dagnn.ReLU(), layer.inputs, {layer.name}, {});
          case 'pool'
            newmodel.addLayer(layer.name, dagnn.Pooling('method', layer.method, 'poolSize', layer.pool, 'stride', layer.stride, 'pad', layer.pad), layer.inputs, {layer.name}, {});
          case 'poolInd2'
            newmodel.addLayer(layer.name, dagnn.PoolingInd('method', layer.method, 'poolSize', layer.pool, 'stride', layer.stride, 'pad', layer.pad), layer.inputs, {layer.name, sprintf('%s%s', layer.name, '_indices'), sprintf('%s%s', 'sizes_pre_', layer.name), sprintf('%s%s', 'sizes_post_', layer.name)}, {});
          case 'unpool'
            newmodel.addLayer(layer.name, dagnn.Unpooling(), {layer.inputs{1}, sprintf('%s%s', layer.ID, '_indices'), sprintf('%s%s', 'sizes_pre_', layer.ID), sprintf('%s%s', 'sizes_post_', layer.ID)}, {layer.name}, {});
          case 'sum'
            newmodel.addLayer(layer.name, dagnn.Sum(), layer.inputs, {layer.name}, {});  
          case 'bnorm'
            newmodel.addLayer(layer.name, dagnn.BatchNorm('numChannels', numel(layer.weights{1})), layer.inputs, {layer.name}, cell_params(l, :));
          case 'dropout'
            newmodel.addLayer(layer.name, dagnn.DropOut('rate', layer.rate), layer.inputs, {layer.name}, {});
          case 'softmax'
            newmodel.addLayer(layer.name, dagnn.SoftMax(), layer.inputs, {layer.name}, {});
          case 'loss'
          case 'loss0'
            newmodel.addLayer(layer.name, dagnn.LossSemantic('weights', true), layer.inputs, {layer.name});
          otherwise
				error('Unknown layer type %s', layer.type) ;
      end
    end
    
    % copy weights
    newmodel.initParams();
    for l=1:size(cell_params, 1)
        if(numel(cell_params{l,:}) > 0)
            for j=1:numel(newmodel.layers(l).paramIndexes)
                 ind = newmodel.layers(l).paramIndexes(j);
                 newmodel.params(ind).value = model.layers{l}.weights{j};
            end
        end
    end
    
    netout = newmodel.saveobj();
    if(nargin == 2)
        save(outputfile, 'netout');
    end
end