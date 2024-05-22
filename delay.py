from cosmic.evolve import Evolve
import warnings

def get_t_pre_ms(M):
    """Get the pre-main sequence evolution timescale for a star of mass M (approximately!)

    Parameters
    ----------
    M : :class:`np.ndarray`
        Mass of the star in solar masses

    Returns
    -------
    t_pre_ms : :class:`np.ndarray`
        Pre-main sequence evolution timescale in Myr
    """
    return 62.8 * M**-2.5

def get_delays(initC):
    """Get the delay times for each companion in the population

    Parameters
    ----------
    initC : :class:`pandas.DataFrame`
        Initial binary population

    Returns
    -------
    delays : :class:`np.ndarray`
        Delay times for each companion in the population
    """
    return get_t_pre_ms(initC["mass_2"].values) - get_t_pre_ms(initC["mass_1"].values)

def delay_companions(initC, delays, pool, BSEdict):
    """Delay the evolution of companions in a binary population due to pre-main sequence evolution timescale
    differences.

    Parameters
    ----------
    initC : :class:`pandas.DataFrame`
        Initial binary population
    delays : :class:`numpy.ndarray`
        Delay time for each companion in the population
    pool : :class:`multiprocessing.Pool`
        Pool of workers for parallel processing
    BSEdict : :class:`dict`
        BSE settings dictionary

    Returns
    -------
    new_initC : :class:`pandas.DataFrame`
        Updated initial binary population with delayed companions
    interacting_binaries : :class:`numpy.ndarray`
        List of binary systems that are interacting after the delay (the problematic ones)
    """
    # create a new initC table
    new_initC = initC.copy()

    # first, evolve stars until delay is reached
    new_initC["tphysf"] = delays

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*initial binary table is being overwritten.*")
        warnings.filterwarnings("ignore", message=".*to a different value than assumed in the mlwind.*")

        # evolve the stars for the delay amount of time
        bpp = Evolve.evolve(initialbinarytable=new_initC, BSEDict=BSEdict, pool=pool)[0]
        updater = bpp.drop_duplicates(subset="bin_num", keep="last")

    interacting_binaries = bpp[(bpp["evol_type"] >= 3) & (bpp["evol_type"] <= 8)]["bin_num"].unique()
    if len(interacting_binaries) > 0:
        warnings.warn((f"Interacting binaries found in the evolved population: {interacting_binaries}. "
                      "This may cause issues with the delay funcionality"))

    columns_to_update = ['porb', 'ecc', 'kstar_1', 'mass_1', 'mass0_1', 'rad_1', 'lum_1',
                         'massc_1', 'radc_1', 'menv_1', 'renv_1', 'omega_spin_1', 'B_1',
                         'bacc_1', 'tacc_1', 'epoch_1', 'tms_1', 'bhspin_1']
    
    new_initC[columns_to_update] = updater[columns_to_update].values
    new_initC["tphysf"] = initC["tphysf"].values
    return new_initC, interacting_binaries