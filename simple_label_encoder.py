def simple_label_encoder(dataframe):
    '''
    Label encode all object-type feature-columns in a dataframe.
    
    Finds all object-type feature-columns and asserts that at least 1 exists.
    The first column is encoded using SciKit-Learn's LabelEncoder() and stored in a 2D array (for concatenation purposes)
    and iteratively goes over the other columns concatenating the new columns to the 2D array.
    All new columns are stored as rows.
    
    Returns the transpose of the array of encoded features.
    
    Needs functionality for seeing if a date-time convertible object exists
    
    PARAMETERS
    ----------
    dataframe: a Pandas dataframe-type object
    
    RETURNS
    -------
    #Dataframe of the new encoded stuff
    #transposed array
    '''

    object_columns = dataframe.dtypes.index[dataframe.dtypes.values == 'O']
    assert has_object_columns(object_columns), 'No Object-type columns found'
    ic(object_columns)

    # Encode the first feature to define array
    le = LabelEncoder()
    le = le.fit_transform(dataframe[object_columns[0]])   # Encode the first column
    encoding = np.array([le], dtype = int)                # Must be 2D for concatenation: `[le]`
    ic(encoding)

    # Want to change this to make sure only > 2 object columns work for this function
    if len(object_columns) > 1:
        for column in object_columns[1:]:                 # idx 0 object-column was already encoded: `object_columns[1:]`
            ic(len(dataframe[column]))
            le = LabelEncoder()
            le = le.fit_transform(dataframe[column])
            ic(len(le))
            temp_arr = np.array([le])                     # Encoded column -> 2D array 
            encoding = np.concatenate((encoding, temp_arr), dtype = int) # Concatenate 2D arrays
            ic(encoding)

    df = pd.DataFrame(encoding.T, columns= object_columns)
    return df
