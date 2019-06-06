# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:14:28 2019

Created by Jared Shenson
https://github.com/jashenson/Multispectral/blob/master/Labelbox/labelboxdl.py


usage: labelboxdl.py [-h] --src SRC [--dest DEST] [-r]

Download Labelbox masks

optional arguments:
  -h, --help   show this help message and exit
  --src SRC    The Labelbox exported CSV
  --dest DEST  Directory to save files in, defaults to script directory
  -r           If included, will overwrite any existing files of the same
               name.

**Install beforehand:
    pip install wget
** Run in terminal to use parser: (with csv in current directory)
    python labelboxdl.py --src export-2019-06-05T17_16_37.771Z.csv


@author: CTLab
George S. Liu
5-20-19

"""

import argparse
import os
import shutil
import csv
import wget
import json

#ID	DataRow ID	Labeled Data	Label	Created By	Project Name	Created At	Updated At	Seconds to Label	External ID	Agreement	Benchmark Agreement	Benchmark ID	Benchmark Reference ID	Dataset Name	Reviews	View Label	Masks
#cjuogkgngcvx008717zvzo405	cjuogj49j06380bqpgg9nv54b	https://storage.googleapis.com/labelbox-193903.appspot.com/cju8qjb1t20p10801sb8fh24p%2Fd5a6d81e-ddac-cb1b-6e40-7074c1d17c45-Artery_arriwhite20_fIRon.png	{"Artery":[{"geometry":[{"x":3,"y":188},{"x":504,"y":217},{"x":1001,"y":226},{"x":1461,"y":214},{"x":1919,"y":190},{"x":1920,"y":806},{"x":1796,"y":835},{"x":1642,"y":868},{"x":1406,"y":887},{"x":943,"y":873},{"x":689,"y":847},{"x":427,"y":834},{"x":369,"y":827},{"x":4,"y":844}]}]}	gliu2@stanford.edu	Multispectral Imaging	1.5557E+12	1.55571E+12	43.55	Artery_arriwhite20_fIRon.png					FreshCadaver001_20190402	[{"id":"cjuoimcgyd9g80818792w3fwr","score":1,"createdAt":"2019-04-19T20:17:32.000Z","createdBy":"jshenson@stanford.edu"}]	https://image-segmentation-v4.labelbox.com?project=cjuog5nqqbeca0987cdpzb7ty&label=cjuogkgngcvx008717zvzo405	{"Artery":"https://faas-gateway.labelbox.com/function/mask-exporter?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsYWJlbElkIjoiY2p1b2drZ25nY3Z4MDA4NzE3enZ6bzQwNSIsImNsYXNzTmFtZSI6IkFydGVyeSIsImlhdCI6MTU1Nzk2MTI0MSwiZXhwIjoxNzE1NjQxMjQxfQ.SdvVAoUsSRfvkIYNtsVKU2yakifGrzwijI8guwAICXI"}
class Label(object):
    def __init__(self, dict):
        super(Label, self).__init__()
        self.id = dict['ID']
        self.datarow_id = dict['DataRow ID']
        if dict['Label'] and dict['Label'] != "Skip":
            self.segments = json.loads(dict['Label'])
        else:
            self.segments = []
        self.filename = dict['External ID']
        self.dataset = dict['Dataset Name']
        if dict['Masks']:
            self.masks = json.loads(dict['Masks'])
        else:
            self.masks = []

    def save_masks(self, overwrite):
        for label, url in self.masks.items():
            fileparts = os.path.splitext(self.filename)
            maskname = f"{fileparts[0]}_{label}Mask{fileparts[1]}"
            print(f"Downloading mask: {maskname}")
            if os.path.exists(maskname):
                print("--> Overwriting...")
                os.remove(maskname)
            wget.download(url, maskname, bar=None)

def parse_csv(export):
    reader = csv.DictReader(export)

    labels = []
    for line in reader:
        labels.append( Label(line) )

    return labels

def datasets_from_labels(labels):
    datasets = []
    for label in labels:
        if label.dataset not in datasets:
            datasets.append(label.dataset)
    return datasets

def labels_in_dataset(labels, dataset):
    ds_labels = []
    for label in labels:
        if label.dataset == dataset:
            ds_labels.append(label)
    return ds_labels

def select_dataset():
    datasets = datasets_from_labels(labels)
    print("Datasets:")
    for idx, d in enumerate(datasets, start=1):
        print(f"  {idx}. {d}")

    cmd = int(input("\nSelect a dataset to download (or enter 0 for all): "))

    if cmd == 0:
        return datasets
    elif cmd <= len(datasets):
        return [datasets[cmd -1]]
    else:
        return [datasets[0]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Labelbox masks')
    parser.add_argument('--src', required=True, type=argparse.FileType('r'), help='The Labelbox exported CSV')
    parser.add_argument('--dest', help='Directory to save files in, defaults to script directory')
    parser.add_argument('-r', action='store_true', help='If included, will overwrite any existing files of the same name.')
    args = parser.parse_args()

    # Parse the CSV
    labels = parse_csv(args.src)

    # Confirm directory exists
    if args.dest and os.path.isdir(args.dest):
        # Set working directory
        os.chdir(args.dest)

    # Prompt user for dataset to download (or all)
    selected = select_dataset()
    for d in selected:
        selected_labels = labels_in_dataset(labels, d)
        for l in selected_labels:
            l.save_masks(args.r)