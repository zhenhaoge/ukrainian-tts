import json
import csv
import librosa

def read_trans(trans_file):

    lines = open(trans_file, 'r').readlines()
    num_lines = len(lines)

    sentences = [() for _ in range(num_lines)]
    for i in range(num_lines):
        line = lines[i]
        parts = line.strip().split()
        id = parts[0]
        text = ' '.join(parts[1:])
        sentences[i] = (id, text)

    return sentences

def parse_fid(fid):
    """parse file id: <fid-seq>_<start-time>_<end-time>"""

    fid_seq, start_time, end_time = fid.split('_')
    start_time = float(start_time)
    end_time = float(end_time)

    return fid_seq, start_time, end_time

def get_dur_from_meta(jsonfiles):
    """get duration (refrence, synthesized) from the meta json files"""

    num_files = len(jsonfiles)
    durs_ref = [0 for _ in range(num_files)]
    durs_syn = [0 for _ in range(num_files)]
    for i in range(num_files):
        with open(jsonfiles[i], 'r') as f:
            meta = json.load(f)
            durs_ref[i] = meta['end-time'] - meta['start-time']
            durs_syn[i] = meta['dur-syn']
    return durs_ref, durs_syn

def get_dur_from_file(wavfiles):

    num_wavfiles = len(wavfiles)
    durs = [0 for _ in range(num_wavfiles)]
    for i, f in enumerate(wavfiles):
        durs[i]= librosa.get_duration(path=f)
    return durs

def dict2mat(d):
  """
  convert dictionary with tuple key to matrix
  """
  key1 = sorted(set([k1 for (k1,k2) in d.keys()]))
  key2 = sorted(set([k2 for (k1,k2) in d.keys()]))
  m = [[0 for _ in range(len(key2))] for _ in range(len(key1))]
  for i, k1 in enumerate(key1):
    for j, k2 in enumerate(key2):
      m[i][j] = d[k1,k2]
  return m, key1, key2

def tuple2csv(tuples, csvname='filename.csv', colname=[], verbose=True):
  with open(csvname, 'w', newline='') as f:
    csv_out = csv.writer(f)
    if len(colname) != 0:
      header = colname
      csv_out.writerow(header)
    for i, tpl in enumerate(tuples):
      csv_out.writerow(list(tpl))
  if verbose:
    print('{} saved!'.format(csvname))

def mat2csv(mat, csvname='filename.csv', colname=[], rowname=[], verbose=True):
  """

  Parameters
  ----------
  mat : list of lists (sublists are rows)
    DESCRIPTION. matrix to be written to csv file
  csvname : string, optional
    DESCRIPTION. The default is 'filename.csv'.
  colname : list, optional
    DESCRIPTION. The default is [].
  rowname : list, optional
    DESCRIPTION. The default is [].
  verbose : boolean, optional
    DESCRIPTION. The default is True.

  Returns
  -------
  None.

  """
  with open(csvname, 'w', newline='') as f:
    csv_out = csv.writer(f)
    if len(colname) != 0:
      if len(rowname) != 0:
        header = [''] + colname
      else:
        header = colname
      csv_out.writerow(header)
    for i, lst in enumerate(mat):
      if len(rowname) != 0:
        row = [rowname[i]] + lst
      else:
        row = lst
      csv_out.writerow(row)
  if verbose:
    print('{} saved!'.format(csvname))

def dl2csv(dict_list, header, csvname='filename.csv', verbose=True):
  """convert dict list to csv, where elements in the list are all dictionaries
     sharing the same keys"""
  with open(csvname, 'w', newline='') as f:
    csv_out= csv.writer(f)
    csv_out.writerow(header)
    for i, dict_ele in enumerate(dict_list):
      row = [dict_ele[k] for k in header]
      csv_out.writerow(row)
  if verbose:
    print('{} saved!'.format(csvname))

def dict2csv(dct, csvname='filename.csv', verbose=True):
  """convert dictionary to csv, where the keys form the header"""
  with open(csvname, 'w', newline='') as f:
    header = list(dct.keys())
    csv_out = csv.writer(f)
    csv_out.writerow(header)
    nitems = len(dct[header[0]])
    for i in range(nitems):
      row = [dct[k][i] for k in header]
      csv_out.writerow(row)
  if verbose:
    print('{} saved!'.format(csvname))

def lst2csv(lst, csvname='filename.csv', colname=[], rowname=[], verbose=True):
  """

  Parameters
  ----------
  lst : TYPE
    DESCRIPTION.
  csvname : TYPE, optional
    DESCRIPTION. The default is 'filename.csv'.
  colname : TYPE, optional
    DESCRIPTION. The default is [].
  rowname : TYPE, optional
    DESCRIPTION. The default is [].
  verbose : TYPE, optional
    DESCRIPTION. The default is True.

  Returns
  -------
  None.

  """
  mat = [[i] for i in lst]
  mat2csv(mat, csvname, colname, rowname, verbose)

def convert_symbol(text, l1, l2, quote='"'):
  """convert symbol l1 to l2 if inside quote"""
  text2 = ''
  inside = False
  for c in text:
    if c == quote:
      inside = not inside
    elif c == l1:
      if inside:
        text2 += l2
      else:
        text2 += l1
    else:
       text2 += c
  return text2

def csv2dict(csvname, delimiter=',', encoding='utf-8'):
  """extract rows in csv file to a dictionary list"""
  lines = open(csvname, 'r', encoding=encoding).readlines()
  header = lines[0].rstrip().split(delimiter)
  lines = lines[1:]
  nlines = len(lines)

  dict_list = [{} for _ in range(nlines)]
  for i, line in enumerate(lines):
    line2 = convert_symbol(line.rstrip(), delimiter, '|')
    items = line2.split(delimiter)
    items = [s.replace('|', delimiter) for s in items]
    dict_list[i] = {k:items[j] for j,k in enumerate(header)}

  return dict_list

def _is_substrs_in_str(s, substrs):
  """
  check if all sub-strings are in string
  """
  for ss in substrs:
    if ss not in s:
      return False
  return True

def extract_flist(flist, keys):
  sublst = []
  for f in flist:
    if _is_substrs_in_str(f, keys):
      sublst.append(f)
  return sublst

def write_flist(flist, lfpath):
  open(lfpath, 'w').write('\n'.join(flist))

def flatten_list(l):
  return [item for sublist in l for item in sublist]     