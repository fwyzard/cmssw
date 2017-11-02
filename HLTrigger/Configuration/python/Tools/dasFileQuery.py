import sys
import os
import json
import das_client

def ca():
  default_ca = os.environ.get("X509_CERT_DIR")
  if not default_ca or not os.path.exists(default_ca):
    default_ca = "/etc/grid-security/certificates"
    if not os.path.exists(default_ca):
      default_ca = ""
  return default_ca


def dasFileQuery(dataset):
  query   = 'dataset dataset=%s' % dataset
  host    = 'https://cmsweb.cern.ch'      # default
  idx     = 0                             # default
  limit   = 0                             # unlimited
  debug   = 0                             # default
  thr     = 300                           # default
  ckey    = das_client.x509()             # default
  cert    = das_client.x509()             # default
  capath  = ca()                          # default
  das_client.check_auth(ckey)
  jsondict = das_client.get_data(host, query, idx, limit, debug, thr, ckey, cert)

  # check if the pattern matches none, many, or one dataset
  if not jsondict['data'] or not jsondict['data'][0]['dataset']:
    sys.stderr.write('Error: the pattern "%s" does not match any dataset\n' % dataset)
    sys.exit(1)
    return []
  elif len(jsondict['data']) > 1:
    sys.stderr.write('Error: the pattern "%s" matches multiple datasets\n' % dataset)
    for d in jsondict['data']:
      sys.stderr.write('    %s\n' % d['dataset'][0]['name'])
    sys.exit(1)
    return []
  else:
    # expand the dataset name
    dataset = jsondict['data'][0]['dataset'][0]['name']
    query = 'file dataset=%s' % dataset
    jsondict = das_client.get_data(host, query, idx, limit, debug, thr, ckey, cert)
    # parse the results in JSON format, and extract the list of files
    files = sorted( f['file'][0]['name'] for f in jsondict['data'] )
    return files

