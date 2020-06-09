# Analysis

## 04/09/2020

[x] Generate movies with timeseries overlays
[o] Automatically annotate artifacts
    [x] Annotate artifacts due to convulsing
    [ ] Annotate recording artifacts
    [x] Stats of artifact times vs good times
[x] Extract connectivity metrics
[x] Plot/visualize connectivity metrics
[x] Run UMAP and visualize data
   [x] Define different states
      [x] mean_lfp
      [x] spectral_state
      [ ] others?
   [x] Apply UMAP with different state definitions


# Package maintainance and enhancements

## 04/15/2020

[o] How should a pipeline look for a new experiment?  
    1. Do this once (or automatically check and do it if not already done):
       1. Create EEGExp object that loads all the metadata from exported files
       2. Run `align_eeg_timestamps` to generate the timestamps file for eeg data
    2. For every analysis:
       1. Create EEGExp object to load metadata
       2. Use `load_*` to load different kinds of data either as numpy array
          or as pandas dataframe and proceed with analysis  
          `load_*` takes a downsampling factor as a parameter. Use it to
          downsample and save lfp; or just load downsampled running and iso

## 04/09/2020

[ ] Integrate movie and lfp classes into eegexp class
[x] eegexp lfp data does not stay memory mapped, but gets loaded
    when the array is reshaped  
    This should either be fixed, or once the data is downsampled
    and filtered, the new array should be saved such that it does
    not have to be loaded, but can be memory mapped. Memory mapping,
    however, is difficult if you are going to use dataframes anyway.  
    Turns out, I can just save the reshaped array after downsampling
    so that it remains memory-mapped at least when loaded as a numpy array.  