function launch_net(imdb, netName, experimentID, opt_GPUID, preNet, varargin)
	% setting up paths	

	run(fullfile(fileparts(mfilename('fullpath')), '../', 'matlab', 'vl_setupnn.m')) ;
	addpath(genpath('networks'));
	addpath(genpath('trainers'));
	
	% initialize GPU
	GPU_ID = opt_GPUID;
	gpuDevice(GPU_ID)

	% create experiment directory and save configuration
	if(~exist(experimentID, 'dir'))
		mkdir(experimentID);
	end

	% we get the actual name of the database
	imdb_name = inputname(1);
	% and the complete net
	netCODE = fileread(['networks/', netName, '.m']);
	% save the imdb_name and the conf of the net
	fd = fopen([experimentID, '/experiment_configuration.txt'], 'w');
	fprintf(fd,' [DataBase = %s]\n*****************\n\n\n%s\n', imdb_name, netCODE);
	fclose(fd);

	% call the net to start the experiment
	input_opts.expDir = experimentID;

	net_handle =  str2func(netName);
	[fnet, infor] = net_handle(imdb, preNet, input_opts, varargin);

	% save net and info
	save([experimentID '/fnet.mat'], 'fnet');
	save([experimentID '/infor.mat'], 'infor');
end
