import subprocess
import glob
import os
import re
import pathlib
import time

import numpy as np
import pandas as pd
import pyarrow.feather as feather

from workbench.utilities import general, temporal, io, constants


def change_plastic_material(og_material_fp, blk_material_fp, line_idx, item_idx, new_name):
    with open(og_material_fp, "r") as fp:
        material_lines = fp.readlines()

    new_line = []
    for n, i in enumerate(material_lines[line_idx].split(" ")):

        if n == item_idx:
            new_line.append(new_name)
        else:
            new_line.append(i)
    new_line = " ".join(new_line) + "\n"
    material_lines[line_idx] = new_line

    with open(blk_material_fp, "w") as fp:
        fp.writelines(material_lines)

def create_black_objects(input_f, output_f):
    cmd = f"echo !xform -m black {input_f} > {output_f}"
    subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)


def build_cmd_oconv(radiance_project_dir, radiance_surface_key, step):
    cmd = ['oconv']

    radiance_surface_dir = os.path.join(radiance_project_dir, f"surface_{radiance_surface_key}")
    oct_dir = os.path.join(radiance_surface_dir, "outputs", "octree")
    io.directory_creator(oct_dir)

    object_material_file = os.path.join(radiance_surface_dir, "model", "scene", "envelope.mat")
    black_object_material_file = os.path.join(radiance_surface_dir, "model", "scene", "envelope.blk")
    change_plastic_material(object_material_file, black_object_material_file, 0 , 2, "black")

    object_file = os.path.join(radiance_surface_dir, "model", "scene", "envelope.rad")
    black_object_file = os.path.join(radiance_surface_dir, "model", "scene", "envelope_black.rad")
    if os.path.exists(black_object_file):
        pass
    else:
        create_black_objects(object_file, black_object_file)


    glazing_material_file = os.path.join(radiance_surface_dir, "model", "aperture", "aperture.mat")
    if os.path.exists(glazing_material_file):
        black_glazing_material_file = os.path.join(radiance_surface_dir, "model", "aperture", "aperture.blk")
        change_plastic_material(glazing_material_file, black_glazing_material_file, 0, 2, "black")

        glazing_file = os.path.join(radiance_surface_dir, "model", "aperture", "aperture.rad")
        black_glazing_file = os.path.join(radiance_surface_dir, "model", "aperture", "aperture_black.rad")

        if os.path.exists(black_glazing_file):
            pass
        else:
            create_black_objects(glazing_file, black_glazing_file)
    else:
        pass

    sun_file = os.path.join(radiance_surface_dir, "model", "scene", "suns.rad")

    if step == "total":
        oct_name = "total.oct"
        output_file = os.path.join(oct_dir, oct_name)
        if os.path.exists(glazing_material_file):
            cmd += [object_material_file, object_file, glazing_material_file, glazing_file]  # , ">", output_file]
        else:
            cmd += [object_material_file, object_file]  # , ">", output_file]

    elif step == "direct":
        oct_name = "direct.oct"
        output_file = os.path.join(oct_dir, oct_name)
        if os.path.exists(glazing_material_file):
            cmd += [black_object_material_file, black_object_file, glazing_material_file,
                    glazing_file]  # , ">", output_file]
        else:
            cmd += [black_object_material_file, black_object_file]

    elif step == "sun":
        oct_name = "sun.oct"
        output_file = os.path.join(oct_dir, oct_name)
        if os.path.exists(sun_file):
            if os.path.exists(glazing_material_file):
                cmd += ["-f", black_object_material_file, black_object_file, sun_file, glazing_material_file,
                        glazing_file]  # , ">", output_file]
            else:
                cmd += ["-f", black_object_material_file, black_object_file, sun_file]

        else:
            print(
                "The file 'suns.rad' is missing. This can be generated from a primiary solar material and then 'rcalc' with a reinhart sky.")
            return FileNotFoundError

    else:
        print("Arg 'step' must be specified as 'total', 'direct', or 'sun'")
        return TypeError

    return cmd, output_file


def create_skyglow(skyglow_file, resolution, dst):
    """_summary_

    Args:
        skyglow_file (str): the tempalte skyglow file
        resolution (int): An integer from 1 to 6 that determines the number of sky subdivisions for the skyglow. 1 is a Tregenza sky.
        dst (str): where to save the new skyglow file
    """
    with open(skyglow_file, "r") as fp:
        skyglow_lines = fp.readlines()

    new_lines = []
    for l in skyglow_lines:
        if "h=r" in l:
            n_l = re.sub("\d+(?=\D*$)", resolution, l)
            new_lines.append(n_l)
        else:
            new_lines.append(l)

    with open(dst, "w") as fp:
        fp.writelines(new_lines)


def build_cmd_rfluxmtx(radiance_project_dir, radiance_surface_key, skyglow_template, step,
                       n_workers=None, rad_params=None):
    cmd = ["rfluxmtx", "-I+"]
    radiance_surface_dir = os.path.join(radiance_project_dir, f"surface_{radiance_surface_key}")
    output_dir = os.path.join(radiance_surface_dir, "outputs", "matrices")
    io.directory_creator(output_dir)

    skyglow_file = os.path.join(radiance_surface_dir, "model", "scene", "skyglow.rad")
    if os.path.exists(skyglow_file):
        pass
    else:
        # TODO investigate why sky resolution cannot be changed from h=r1
        sky_resolution = 1
        create_skyglow(skyglow_template, str(sky_resolution), skyglow_file)

    oct_dir = os.path.join(radiance_surface_dir, "outputs", "octree")
    if step == 'total':
        octree_file = os.path.join(oct_dir, "total.oct")
        output_file = os.path.join(output_dir, 'total_illum.mtx')
    elif step == 'direct':
        octree_file = os.path.join(oct_dir, "direct.oct")
        output_file = os.path.join(output_dir, 'direct_illum.mtx')
    else:
        print("Arg 'step' must be specified as 'total' or 'direct'")
        return TypeError

    grid_file = glob.glob(os.path.join(radiance_surface_dir, "model", "grid", "*.pts"))[0]
    line_count = int(grid_file.split("_")[-1].split("s")[0])
    # lines in grid file
    # line_count = utils.count_lines(grid_file)
    cmd += ["-y", f"{int(line_count)}"]

    # number of workers
    if n_workers is None:
        n_workers = os.cpu_count() - 1
    cmd += ["-n", f"{int(n_workers)}"]

    # rad_params
    if rad_params is None:
        rad_params = "-lw 0.0001 -ab 5 -ad 10000"
    cmd += rad_params.split(" ")

    # skyglow
    cmd += ["-", f"{skyglow_file}"]

    # sender and octree
    cmd += ["-i", f"{octree_file}", "<", f"{grid_file}"]  # , ">", f"{output_file}"]

    return cmd, output_file, grid_file


def build_cmd_epw2wea(radiance_project_dir, radiance_surface_key, input_epw):
    input_epw = pathlib.Path(input_epw)
    radiance_surface_dir = os.path.join(radiance_project_dir, f"surface_{radiance_surface_key}")
    wea_name = input_epw.name.replace(".epw", ".wea")

    output_wea = os.path.join(radiance_surface_dir, 'model', wea_name)

    cmd = ['epw2wea']

    cmd += [input_epw]
    cmd += [output_wea]
    return cmd, output_wea


def build_cmd_gendaymtx(radiance_project_dir, radiance_surface_key, wea_file, step):
    """_summary_
    example: gendaymtx -m 1 assets/NYC.wea > skyVectors/NYC.smx
    sun-coe example: gendaymtx -5 0.533 -d -m 6 assets/NYC.wea > skyVectors/NYCsunM6.smx
    Args:
        radiance_project_dir (_type_): _description_
        radiance_surface_key (_type_): _description_
        wea_file (_type_): _description_
        step (_type_): _description_

    Returns:
        _type_: _description_
    """
    cmd = ['gendaymtx']

    radiance_surface_dir = os.path.join(radiance_project_dir, f"surface_{radiance_surface_key}")
    output_dir = os.path.join(radiance_surface_dir, "outputs", "matrices")

    if step == 'total':
        output_file = os.path.join(output_dir, 'sky_total.smx')
        cmd += ['-m', '1']
    elif step == 'direct':
        output_file = os.path.join(output_dir, 'sky_direct.smx')
        cmd += ['-m', '1', '-d']
    elif step == 'sun':
        output_file = os.path.join(output_dir, 'sky_sun.smx')
        cmd += ['-5', '0.533', '-d', '-m', '6']
    else:
        print("Arg 'step' must be specified as 'total', 'direct', or 'sun'")
        return TypeError

    cmd += [wea_file]

    return cmd, output_file


def build_cmd_dctimestep(radiance_project_dir, radiance_surface_key, step):
    """_summary_
    example: dctimestep matrices/cds/cdsDDS.mtx skyVectors/NYCsunM6.smx
    Args:
        radiance_project_dir (_type_): _description_
        radiance_surface_key (_type_): _description_
    """

    cmd = ['dctimestep']
    radiance_surface_dir = os.path.join(radiance_project_dir, f"surface_{radiance_surface_key}")
    matrices_dir = os.path.join(radiance_surface_dir, "outputs", "matrices")

    if step == 'total':
        input_matrix = os.path.join(matrices_dir, 'total_illum.mtx')
        input_sky = os.path.join(matrices_dir, 'sky_total.smx')
    elif step == 'direct':
        input_matrix = os.path.join(matrices_dir, 'direct_illum.mtx')
        input_sky = os.path.join(matrices_dir, 'sky_direct.smx')
    elif step == 'sun':
        input_matrix = os.path.join(matrices_dir, 'sun_illum.mtx')
        input_sky = os.path.join(matrices_dir, 'sky_sun.smx')
    else:
        print("Arg 'step' must be specified as 'total', 'direct', or 'sun'")
        return TypeError

    cmd += [input_matrix, input_sky]

    return cmd


def build_cmd_rmtxop(radiance_project_dir, radiance_surface_key, step):
    """_summary_
    The three numbers specified after -c convert the irradiance values
    to illuminance [lux] by scaling and combining the result according to the photopic efficiency function.
    The conversion from lux to W/m2 is applied at the end of the workflow when all result .ill are combined.

    example: rmtxop -fa -t -c 47.4 119.9 11.6 - > results/dcDDS/dc/annualR.ill
    Args:
        radiance_project_dir (_type_): _description_
        radiance_surface_key (_type_): _description_
    """

    cmd = ['rmtxop']
    cmd += ["-fa", "-t", "-c", "47.4", "119.9", "11.6", "-"]
    radiance_surface_dir = os.path.join(radiance_project_dir, f"surface_{radiance_surface_key}")
    output_dir = os.path.join(radiance_surface_dir, "outputs", "results")
    io.directory_creator(output_dir)

    if step == 'total':
        output_file = os.path.join(output_dir, 'result_total.ill')
    elif step == 'direct':
        output_file = os.path.join(output_dir, 'result_direct.ill')
    elif step == 'sun':
        output_file = os.path.join(output_dir, 'result_sun.ill')
    else:
        print("Arg 'step' must be specified as 'total', 'direct', or 'sun'")
        return TypeError

    return cmd, output_file


def create_primitive_sun(radiance_project_dir, radiance_surface_key):
    radiance_surface_dir = os.path.join(radiance_project_dir, f"surface_{radiance_surface_key}")
    scene_dir = os.path.join(radiance_surface_dir, "model", "scene")
    output_file = os.path.join(scene_dir, "suns.rad")
    write_line = "void light solar 0 0 3 1e6 1e6 1e6\n"

    with open(output_file, 'w') as file:
        file.write(write_line)

    return output_file


def build_cmd_cnt():
    cmd = ["cnt"]

    cmd += ['5165']

    return cmd


def build_cmd_rcalc(cal='reinsrc.cal'):
    cmd = ['rcalc']

    cmd += ['-e', 'MF:6', '-f', cal, '-e', 'Rbin=recno', '-o', r"'solar source sun 0 0 4 ${Dx} ${Dy} ${Dz} 0.533'"]

    return cmd


def build_cmd_rcontrib(radiance_project_dir, radiance_surface_key, cal="reinhart.cal", n_workers=None, rad_params=None):
    cmd = ["rcontrib", "-I+", "-ab", "1"]
    radiance_surface_dir = os.path.join(radiance_project_dir, f"surface_{radiance_surface_key}")
    output_dir = os.path.join(radiance_surface_dir, "outputs", "matrices")
    io.directory_creator(output_dir)
    output_file = os.path.join(output_dir, 'sun_illum.mtx')

    sun_oct = os.path.join(radiance_surface_dir, "outputs", "octree", "sun.oct")
    # rcontrib -I+ -ab 1 -y 100 -n 16 -ad 256 -lw 1.0e-3 -dc 1 -dt 0 -dj 0 -faf -e MF:6 -f reinhart.cal -b rbin -bn Nrbins -m solar octrees/sunCoefficientsDDS.oct < points.txt > matrices/cds/cdsDDS.mtx

    grid_file = glob.glob(os.path.join(radiance_surface_dir, "model", "grid", "*.pts"))[0]
    line_count = int(grid_file.split("_")[-1].split("s")[0])
    # lines in grid file
    # line_count = utils.count_lines(grid_file)
    cmd += ["-y", f"{int(line_count)}"]

    # number of workers
    if n_workers is None:
        n_workers = os.cpu_count() - 1
    cmd += ["-n", f"{int(n_workers)}"]

    # rad_params
    if rad_params is None:
        rad_params = "-ad 256 -lw 1.0e-3 -dc 1 -dt 0 -dj 0"
    cmd += rad_params.split(" ")

    cmd += ["-faf", "-e", "MF:6", "-f", cal, "-b", "rbin", "-bn", "Nrbins", "-m", "solar", sun_oct]

    return cmd, output_file, grid_file


def run_2phase_dds(project, year=2099):
    radiance_surface_key = project.analysis_active_surface

    radiance_project_dir = project.RADIANCE_DIR
    scenario_tmy = project.TMY_FILE

    # analysis_period = project.analysis_period
    skyglow_template_file = project.skyglow_template
    n_workers = project.irradiance_n_workers
    rflux_rad_params = project.irradiance_radiance_param_rflux
    rcontrib_rad_params = project.irradiance_radiance_param_rcontrib


    ### Part 0
    print(f" - Running 2-Phase DDS with {n_workers} workers")
    print(f" - Current surface is {radiance_surface_key}")
    start_time = time.time()
    ## epw2wea with filtering
    # build command
    print(" - Initializing the opening weather file.")
    if pathlib.Path(scenario_tmy).suffix == ".epw":
        cmd_epw2wea, output_wea = build_cmd_epw2wea(radiance_project_dir, radiance_surface_key, scenario_tmy)
        # run command
        proc = subprocess.run(cmd_epw2wea, check=True, input=None, stdout=subprocess.PIPE)
        # filter wea
        # temporal.filter_wea(output_wea, year, analysis_period)
    else:
        output_wea = scenario_tmy

    ### [Part 1: Total Irradiance, Part 2: Direct Irradiance]
    for n, step in enumerate(["total", "direct"]):
        print(f" - Starting Part {n + 1} ({step}).")
        ## oconv
        print("     - oconv")
        cmd_oconv, output = build_cmd_oconv(radiance_project_dir, radiance_surface_key, step)
        with open(output, 'w') as fp:
            proc = subprocess.run(cmd_oconv, check=True, input=None, stdout=fp)

        ## rfluxmtx
        print("     - rfluxmtx")

        cmd_rfluxmtx, output_file, input_file = build_cmd_rfluxmtx(radiance_project_dir, radiance_surface_key,
                                                                   skyglow_template_file, step,
                                                                   n_workers=n_workers, rad_params=rflux_rad_params)
        # print(" ".join(cmd_rfluxmtx) + " > " + output_file)
        with open(input_file, "rb") as fp:
            # TODO work on a way to bring timeout back
            proc = subprocess.run(cmd_rfluxmtx, check=True, stdin=fp,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)#, timeout=5)
        with open(output_file, "wb") as fp:
            fp.write(proc.stdout)


        ## gendaymtx
        print("     - gendaymtx")
        cmd_gendaymtx, output_file = build_cmd_gendaymtx(radiance_project_dir, radiance_surface_key,
                                                         output_wea, step)
        with open(output_file, 'w') as file:
            proc = subprocess.run(cmd_gendaymtx, stderr=subprocess.PIPE, stdout=file, check=True)

        ## dctimestep and rmtxop
        print("     - dctimestep | rmtxop")
        # First command: dctimestep matrices/dc/illum.mtx skyVectors/NYC.smx
        cmd_dctimestep = build_cmd_dctimestep(radiance_project_dir, radiance_surface_key, step)
        proc_dctimestep = subprocess.Popen(cmd_dctimestep, stdout=subprocess.PIPE)
        # Second command: rmtxop -fa -t -c 47.4 119.9 11.6 -
        cmd_rmtxop, output_file = build_cmd_rmtxop(radiance_project_dir, radiance_surface_key, step)
        proc_rmtxop = subprocess.Popen(cmd_rmtxop, stdin=proc_dctimestep.stdout,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
        with open(output_file, 'wb') as file:
            file.write(proc_rmtxop.communicate()[0])

    ### Part 3: Sun Coefficients
    step = "sun"
    print(f" - Starting Part {3} ({step}).")

    ## Create solar discs sun file
    # Create a primitive sun
    print("     - create_primitive_sun")
    prim_sun_file = create_primitive_sun(radiance_project_dir, radiance_surface_key)
    # build comand for count
    cmd_cnt = build_cmd_cnt()
    # build rcalc command
    print("     - rcalc")
    cmd_rcalc = build_cmd_rcalc()
    # merge and point to file
    full_cmd = " ".join(cmd_cnt) + " | " + " ".join(cmd_rcalc) + f" >> {prim_sun_file}"
    # run
    subprocess.run(full_cmd, shell=True)

    ## Run oconv
    print("     - oconv")
    cmd_oconv, output = build_cmd_oconv(radiance_project_dir, radiance_surface_key, step)
    with open(output, 'w') as fp:
        proc = subprocess.run(cmd_oconv, check=True, input=None, stdout=fp)

    ## Run rcontrib
    print("     - rcontrib")
    cmd_rcontrib, output_file, input_file = build_cmd_rcontrib(radiance_project_dir, radiance_surface_key,
                                                               n_workers=n_workers,
                                                               rad_params=rcontrib_rad_params)
    with open(input_file, "rb") as fp:
        proc = subprocess.run(cmd_rcontrib, check=True, stdin=fp,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)#, timeout=5)
    with open(output_file, "wb") as fp:
        fp.write(proc.stdout)

    ## Run gendaymtx
    print("     - gendaymtx")
    cmd_gendaymtx, output_file = build_cmd_gendaymtx(radiance_project_dir, radiance_surface_key,
                                                     output_wea, step)
    with open(output_file, 'w') as file:
        subprocess.run(cmd_gendaymtx, stderr=subprocess.PIPE, stdout=file, check=True)

    ## dctimestep and rmtxop for total irradiance
    print("     - dctimestep | rmtxop")
    # First command: dctimestep matrices/dc/illum.mtx skyVectors/NYC.smx
    cmd_dctimestep = build_cmd_dctimestep(radiance_project_dir, radiance_surface_key, step)
    proc_dctimestep = subprocess.Popen(cmd_dctimestep, stdout=subprocess.PIPE)
    # Second command: rmtxop -fa -t -c 47.4 119.9 11.6 -
    cmd_rmtxop, output_file = build_cmd_rmtxop(radiance_project_dir, radiance_surface_key, step)
    proc_rmtxop = subprocess.Popen(cmd_rmtxop, stdin=proc_dctimestep.stdout, stdout=subprocess.PIPE)
    with open(output_file, 'wb') as file:
        file.write(proc_rmtxop.communicate()[0])
    total_time = round(time.time() - start_time,2)
    project.log(total_time, "irradiance")
    return None

def ill_to_df(project):
    radiance_project_dir = project.RADIANCE_DIR
    radiance_surface_key = project.analysis_active_surface
    lux_to_wattm2 = constants.lux_to_wattm2
    radiance_surface_dir = os.path.join(radiance_project_dir, f"surface_{radiance_surface_key}")
    output_dir = os.path.join(radiance_surface_dir, "outputs", "results")

    filepath_total = os.path.join(output_dir, 'result_total.ill')
    filepath_direct = os.path.join(output_dir, 'result_direct.ill')
    filepath_sun = os.path.join(output_dir, 'result_sun.ill')

    df_total = io.read_ill(filepath_total)
    df_direct = io.read_ill(filepath_direct)
    df_sun = io.read_ill(filepath_sun)

    indirect_illuminance = df_total - df_direct

    direct = df_sun * lux_to_wattm2
    direct = direct.astype("float").round(2)
    diffuse = indirect_illuminance * lux_to_wattm2
    diffuse = diffuse.astype("float").round(2)
    diffuse = pd.DataFrame(np.where(diffuse < 0, direct*0.01, diffuse))
    return direct, diffuse


def save_irradiance_results(project):
    print(" - Saving Irradiance results")
    direct, diffuse = ill_to_df(project)

    start_time = time.time()
    project.get_irradiance_results()
    feather.write_feather(direct, project.DIRECT_IRRAD_FILE, compression='lz4')
    end_time = time.time()
    print(f"    - Direct sensor data saved in compressed format, time={round(end_time-start_time,0)}-seconds.")

    start_time = time.time()
    feather.write_feather(diffuse, project.DIFFUSE_IRRAD_FILE, compression='lz4')
    end_time = time.time()
    print(f"    - Diffuse sensor data saved in compressed format, time={round(end_time-start_time,0)}-seconds.")