import csv
import os
import time
import datetime

base_names = ['SWI', 'bodemvocht', 'bodemberging']

for base_name in base_names:
    print 'base_name: %s' % base_name
    path = "%s_csv/" % base_name  # insert the path to the directory of interest
    dirList=os.listdir(path)

    for fname in dirList:
        if fname.endswith('.csv'):
            print 'processing %s...' % fname
            c = csv.reader(open(path+fname, 'rb'))

            d = {}
            has_data = False
            for row in c:
                # get rid of duplicates
                d[row[0]] = row[1]
                if row[1] and row[1] != 'False':
                    has_data = True
            if has_data:
                data = d.items()
                data.sort()
                out_rows = []
                for dt_str, value in data:
                    dt = datetime.datetime.strptime(dt_str, '%Y%m%d')
                    dt_ms = time.mktime(dt.timetuple())
                    out_rows.append('[%f, %s]' % (dt_ms, value))

                output_filename = fname.replace('.csv', '.json')
                out = open(path+output_filename, 'w')

                prepend = \
"""
{
    "data": [
        {
            "bars": {
                "align": "center",
                "barWidth": 86400000.0,
                "show": true
            },
            "color": "blue",
            "data": [
"""
                # inbetween: [<timestamp in ms>, <value>], separated by commas
                postpend = \
"""
            ],
            "label": "%s"
        }
    ],
    "x_label": null,
    "y_label": "eenheid"
}
"""
                out.write(prepend)
                out.write(', '.join(out_rows))
                out.write(postpend % fname.replace('.csv', ''))

                out.close()
                print '- written %s' % output_filename
