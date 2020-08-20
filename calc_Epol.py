import os
import sys
import numpy as np
from pyscf.data.nist import BOHR

ang2bohr = 1.0/BOHR
nm2ang = 10.0
au2kcal = 627.5094737775374


def load_mol2(fname):
    '''Load a TRIPOS mol2 file from disk.'''
    f = open(fname, 'r')
    atoms = f.read().split('@<TRIPOS>')[2].split('\n')
    natom = len(atoms[1:-1])

    atm_list = []
    q_tot = 0.0

    for ia in range(natom):
        words = atoms[ia+1].split()
        x = float(words[2])*ang2bohr
        y = float(words[3])*ang2bohr
        z = float(words[4])*ang2bohr
        symbol = words[5].split('.')[0]
        q_tot += float(words[-1])
        atm_list.append([symbol, (x, y, z)])

    return atm_list, round(q_tot)


def parmed_load_pqr(fname):
    import parmed

    if not os.path.isfile(fname):
        raise Exception("FN(%s) cannot be opened" % fname)

    struct = parmed.load_file(fname, structure=True)
    # length unit of parmed : A
    # A-->Bohr
    xyz_list = struct.get_coordinates(0)*ang2bohr

    q_list = []
    atm_list = []
    for ia, atom in enumerate(struct.atoms):
        q_list.append(atom.charge)
        x, y, z = xyz_list[ia]
        if atom.name.startswith('Na'):
            atom.atomic_number = 11
        elif atom.name.startswith('Mg'):
            atom.atomic_number = 12
        elif atom.name.startswith('Ca'):
            atom.atomic_number = 20
        elif atom.name.startswith('Zn'):
            atom.atomic_number = 30
        atm_list.append([atom.atomic_number, (x, y, z)])

    q_list = np.array(q_list)

    return atm_list, xyz_list, q_list


def mol_build(fname_mol2, basis):
    """
    fname_mol2: string : filename
    basis: string: basis_set name
    """
    from pyscf import gto

    # XYZ in Bohr
    atm_list, qm_charge = load_mol2(fname_mol2)

    mol = gto.Mole()
    mol.basis = basis
    mol.atom = atm_list
    mol.charge = qm_charge
    mol.unit = 'Bohr'
    mol.verbose = 0
    mol.build()

    return mol


def run_scf(mol):
    from pyscf import scf
    # RHF
    mf_rhf = scf.RHF(mol)
    mf_rhf.run()
    #print('DONE RHF', mf_rhf.e_tot)

    return mf_rhf


def run_scf_solvent(mol, solvent_radii, solvent_lebedev_order):
    from pyscf import scf, solvent

    mf = scf.RHF(mol)
    mf = solvent.ddCOSMO(mf)
    mf.radii_table = solvent_radii
    mf.lebedev_order = solvent_lebedev_order
    mf.run()
    #print('DONE RHF/SOLVENT', mf.e_tot)

    return mf


def run_scf_qmmm(mol, fname_pqr):
    from pyscf import scf, qmmm

    mm_atm_list, mm_xyz_list, mm_q_list = parmed_load_pqr(fname_pqr)

    mf = scf.RHF(mol)
    mf_qmmm = qmmm.mm_charge(mf, mm_xyz_list, mm_q_list)
    mf_qmmm.run()
    #print('DONE RHF/MM', mf_qmmm.e_tot)

    return mf_qmmm


def run_scf_qmmm_solvent(mol, fname_pqr, solvent_radii, solvent_lebedev_order):
    from pyscf import scf, qmmm  # , solvent

    import ddcosmo_qmmm
    """
    There is one key difference between pyscf.solvent.ddcosmo and ddcosmo_qmmm.
    The former (pyscf) generates the grid points of both QM and MM regions, 
    which are used to estimate the electronic density.
    'ddcosmo_qmmm' does not generate the grid points in the MM region.
    """
    mm_atm_list, mm_xyz_list, mm_q_list = parmed_load_pqr(fname_pqr)
    mm_mol = qmmm.create_mm_mol(mm_atm_list, mm_q_list, unit='Bohr')

    qmmm_sol = ddcosmo_qmmm.DDCOSMO(mol, mm_mol)
    #wat_radius = 1.4*ang2bohr
    #qmmm_sol.radii_table = radii.VDW + wat_radius
    qmmm_sol.radii_table = solvent_radii
    qmmm_sol.lebedev_order = solvent_lebedev_order

    # ddCOSMO-QMMM-SCF
    mf = mol.RHF()
    mf = mf.QMMM(mm_xyz_list, mm_q_list)
    mf = ddcosmo_qmmm.ddcosmo_for_scf(mf, qmmm_sol)  # mf.DDCOSMO(qmmm_sol)
    mf.run(verbose=0)
    #print('DONE RHF/MM/SOLVENT', mf.e_tot)

    return mf


def calc_epol(mf_qmmm, mf_qm):
    '''
    Epol = <\psi_qmmm| \hat{H}_qmmm|\psi_qmmm> - <\psi_qm| \hat{H}_\qmmm|\psi_qm>
    '''
    dm_qm = mf_qm.make_rdm1()
    epol = mf_qmmm.e_tot - mf_qmmm.energy_tot(dm_qm)

    return epol


def run_pyscf(fname_mol2, fname_pqr, fname_out, options):
    import time

    print('fname_mol2', fname_mol2)
    print('fname_pqr', fname_pqr)
    print('fname_out', fname_out)

    f_out = open(fname_out, 'w', 1)

    mol = mol_build(fname_mol2, options['basis'])

    # QM
    start = time.time()
    mf_qm = run_scf(mol)
    end = time.time()
    f_out.write('QM:Elapsed Time ' + '%12.6f' % (end-start) + '\n')
    start = end

    # QM/MM
    if True:
        start = time.time()
        mf_ref = run_scf_qmmm(mol, fname_pqr)
        end = time.time()
        f_out.write('QMMM:Elapsed Time ' + '%12.6f' % (end-start) + '\n')
        Epol_QMMM = calc_epol(mf_ref, mf_qm)
        f_out.write('@Epol_IN_PROTEIN ' + '%14.6f' %
                    (Epol_QMMM*au2kcal) + '\n')

    # QM/Solvent
    if True:
        start = time.time()
        mf_ref = run_scf_solvent(mol,
                                 options['solvent_radii'],
                                 options['solvent_lebedev_order'])
        end = time.time()
        f_out.write('QM/Solvent:Elapsed Time ' + '%12.6f' % (end-start) + '\n')
        # Solvent Effect on Epol
        Epol_QM_Solv = calc_epol(mf_ref, mf_qm)
        f_out.write('@Epol_IN_SOLV' + '%14.6f' % (Epol_QM_Solv*au2kcal) + '\n')

    # QM/MM/Solvent
    if True:
        start = time.time()
        mf_ref = run_scf_qmmm_solvent(mol, fname_pqr,
                                      options['solvent_radii'],
                                      options['solvent_lebedev_order'])
        end = time.time()
        f_out.write('QM/MM/Solvent:Elapsed Time ' + '%12.6f' %
                    (end-start) + '\n')

        Epol_QMMM_Solv = calc_epol(mf_ref, mf_qm)
        f_out.write('@Epol_IN_PROT_SOLV' + '%14.6f' %
                    (Epol_QMMM_Solv*au2kcal) + '\n')


if __name__ == '__main__':
    from pyscf.data import radii

    fname_mol2 = 'minimized_ligand.mol2'
    fname_pqr = 'receptor.pqr'
    fname_out = '1bzc.log'

    options = {}
    options['basis'] = '6-311g**'
    """
    wat_radius = 1.4*ang2bohr
    options['solvent_radii'] = radii.VDW + wat_radius
    if we use the above option, 
    then the solvent effect on the polarization energy is negligible.
    """
    options['solvent_radii'] = radii.VDW*1.2
    options['solvent_lebedev_order'] = 7
    run_pyscf(fname_mol2, fname_pqr, fname_out, options)
