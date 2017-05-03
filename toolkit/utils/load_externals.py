import sys, os

external_libs = {'Cleverhans v1.0.0': "externals/cleverhans"}

project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

for lib_name, lib_path in external_libs.iteritems():
    lib_path = os.path.join(project_path, lib_path)
    if os.listdir(lib_path) == []:
        cmd = "git submodule update --init --recursive"
        print("Fetching external libraries...")
        os.system(cmd)

    sys.path.append(lib_path)
    print("Located %s" % lib_name)

# print (sys.path)
