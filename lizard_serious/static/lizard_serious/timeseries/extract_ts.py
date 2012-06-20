# Extract timeseries from a point in a bunch of ascii files.
from asc import AscGrid
import os
from subprocess import call
import csv
import pickle

import os.path

base_names = ["SWI", 'bodemvocht', 'bodemberging']

for base_name in base_names:
    print '-----------------------------------'
    print 'base_name: %s' % base_name
    path = "%s/" % base_name  # insert the path to the directory of interest
    path_result = "%s_csv/" % base_name
    dirList=os.listdir(path)

    collect_coordinates = []
    for x in xrange(0, 1000, 100):
        for y in xrange(0, 1200, 100):
            collect_coordinates.append((x, y))

    counter = 0

    for fname in dirList:
        counter += 1
        if fname.endswith('.asc'):
            print '%s (%d of %d)' % (fname, counter, len(dirList))

            # Speeds up when running multiple times
            pickle_name = path_result+fname+'.pickle'
            try:
                print ' - try reading pickle for %s' % fname
                pickle_file = open(pickle_name, 'r')
                grid = pickle.load(pickle_file)
                pickle_file.close()
            except:
                print ' - reading %s' % fname
                f = open(path+fname, 'r')
                grid = AscGrid(f)
                f.close()
                pickle_file = open(pickle_name, 'w')
                pickle.dump(grid, pickle_file)
                pickle_file.close()

            for x, y in collect_coordinates:
                output_fname = '%s%s_%d_%d.csv' % (path_result, base_name, x, y)
                timestamp = fname.replace(
                    'NEO_DRYMON_', '').replace('SWI_', '').replace(
                    'berging_', '').replace('bodemvocht_', '').replace('.asc', '')
                value = grid[x, y]
                # if value:
                #     print x, y, value
                c = csv.writer(open(output_fname, 'a'))
                c.writerow([timestamp, value])
