def prepare_sites_dict(path_to_data, 
                       sites_dict_file=os.path.join(PATH_TO_DATA, 'sites_dict.pkl'),
                       inds_dict_file=os.path.join(PATH_TO_DATA, 'ind_to_sites_dict.pkl'),
                       refresh=False,
                       return_inds_dict=False):
    """Func to get dictionaries for converting site's name to it's index.
        If dictionary for data in PATH_TO_DATA has already been compiled, 
        functions just pickle dict out of files.
    """
    def get_dict():
        full_df = pd.DataFrame(columns=['site']) # UPD deleted 'timestamp'
        for file in tqdm(glob(path_to_data + '/*'), desc='Preparing sites dict...'):
            temp_df = pd.read_csv(file, usecols=['site'])  # UPD: added usecols
            full_df = full_df.append(temp_df, ignore_index=True)

        sites_freq_list = sorted(Counter(full_df.site).items(), 
                                 key=lambda x: x[1], reverse=True)
        sites_dict = dict((s, [i, freq]) for i, (s, freq) in enumerate(sites_freq_list, 1))
        if return_inds_dict:
            ind_to_sites_dict = dict((val[0], key) for key, val in sites_dict.items())
            ind_to_sites_dict[0] = 'no_site'
        else:
            ind_to_sites_dict = {}
        
        # Save dict to file
        with open(sites_dict_file, 'wb') as fout:
            pickle.dump(sites_dict, fout)
        if return_inds_dict:
            with open(inds_dict_file, 'wb') as fout:
                pickle.dump(ind_to_sites_dict, fout)
            
        return sites_dict, ind_to_sites_dict
    
    try:
        with open(sites_dict_file, 'rb') as input_file:
            sites_dict = pickle.load(input_file)
        if return_inds_dict:    
            with open(inds_dict_file, 'rb') as input_file:
                ind_to_sites_dict = pickle.load(input_file)
            
    except FileNotFoundError:
        sites_dict, ind_to_sites_dict = get_dict()
        
    if refresh:
        sites_dict, ind_to_sites_dict = get_dict()
        
    return sites_dict, ind_to_sites_dict
