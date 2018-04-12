import numpy as np
import pickle

with open('../../test/backprop_array.hpp', mode='w') as f:
    f.write("#pragma once\n\n")
    with open('../../data/Conv_deriv.pkl', 'rb') as fr:
        dcnv = pickle.load(fr)
        f.write("float dconv_raw[] = {")
        for i, xsss in enumerate(dcnv):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('../../data/Affine1_deriv.pkl', 'rb') as fr:
        daffine1 = pickle.load(fr)
        f.write("float daffine1_raw[] = {")
        for i, xsss in enumerate(daffine1):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('../../data/Pool_deriv.pkl', 'rb') as fr:
        dpool1 = pickle.load(fr)
        f.write("float dpool1_raw[] = {")
        for i, xsss in enumerate(dpool1):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('../../data/Relu1_deriv.pkl', 'rb') as fr:
        drelu1 = pickle.load(fr)
        f.write("float drelu1_raw[] = {")
        for i, xsss in enumerate(drelu1):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('../../data/Affine2_deriv.pkl', 'rb') as fr:
        daffine2 = pickle.load(fr)
        f.write("float daffine2_raw[] = {")
        for i, xs in enumerate(daffine2):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('../../data/Relu2_deriv.pkl', 'rb') as fr:
        drelu2 = pickle.load(fr)
        f.write("float drelu2_raw[] = {")
        for i, xs in enumerate(drelu2):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('../../data/Lastlayer_deriv.pkl', 'rb') as fr:
        last = pickle.load(fr)
        f.write("float last_raw[] = {")
        for i, xs in enumerate(last):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('../../data/Relu1.pkl', 'rb') as fr:
        prelu1 = pickle.load(fr)
        f.write("float prelu1_raw[] = {")
        for i, xsss in enumerate(prelu1):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('../../data/Relu2.pkl', 'rb') as fr:
        prelu2 = pickle.load(fr)
        f.write("float prelu2_raw[] = {")
        for i, xs in enumerate(prelu2):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('../../data/Pool1.pkl', 'rb') as fr:
        ppool1 = pickle.load(fr)
        f.write("float ppool1_raw[] = {")
        for i, xsss in enumerate(ppool1):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('../../data/x.pkl', 'rb') as fr:
        px = pickle.load(fr)
        f.write("float px_raw[] = {")
        for i, xsss in enumerate(px):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('../../data/Conv1.pkl', 'rb') as fr:
        pconv1 = pickle.load(fr)
        f.write("float pconv1_raw[] = {")
        for i, xsss in enumerate(pconv1):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
