import os.path as op
import numpy as np
import nibabel as nib
from scipy import linalg

import mne
from mne.io.constants import FIFF

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
raw_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
trans_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
raw = mne.io.read_raw_fif(raw_fname)
trans = mne.read_trans(trans_fname)
src = mne.read_source_spaces(op.join(subjects_dir, 'sample', 'bem',
                                     'sample-oct-6-src.fif'))

# Load the T1 file and change the header information to the correct units
t1w = nib.load(op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz'))
t1w = nib.Nifti1Image(t1w.dataobj, t1w.affine)
t1w.header['xyzt_units'] = np.array(10, dtype='uint8')
t1_mgh = nib.MGHImage(t1w.dataobj, t1w.affine)
fig = mne.viz.plot_alignment(raw.info, trans=trans, subject='sample',
                             subjects_dir=subjects_dir, surfaces='head-dense',
                             show_axes=True, dig=True, eeg=[], meg='sensors',
                             coord_frame='meg', mri_fiducials='estimated')
mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))
print('Distance from head origin to MEG origin: %0.1f mm'
      % (1000 * np.linalg.norm(raw.info['dev_head_t']['trans'][:3, 3])))
print('Distance from head origin to MRI origin: %0.1f mm'
      % (1000 * np.linalg.norm(trans['trans'][:3, 3])))
dists = mne.dig_mri_distances(raw.info, trans, 'sample',
                              subjects_dir=subjects_dir)
print('Distance from %s digitized points to head surface: %0.1f mm'
      % (len(dists), 1000 * np.mean(dists)))