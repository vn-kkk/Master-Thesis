module tov_module
    implicit none
    
    real*8, parameter :: pi = 3.1415926535897932384626433832795d0
    real*8, parameter :: xmsun = 1.4766d0 ! Mass of sun in km
    
    ! Module-level variables to hold the EOS table
    real*8, allocatable, dimension(:) :: eos_p, eos_e
    integer :: n_tab

    ! CRITICAL FIX: Tell f2py only to expose the main interface to Python
    public :: tov_mrl_vector
    private :: solve_single_star, get_derivatives, get_eng, get_vs

contains

    ! --- Main Vectorized Interface for Python ---
    subroutine tov_mrl_vector(e_in, p_in, n_eos, pc_in, n_pc, m_out, r_out, lam_out)
        !f2py intent(in) :: e_in, p_in, pc_in
        !f2py intent(hide) :: n_eos = len(e_in), n_pc = len(pc_in)
        !f2py intent(out) :: m_out, r_out, lam_out
        
        integer, intent(in) :: n_eos, n_pc
        real*8, intent(in), dimension(n_eos) :: e_in, p_in
        real*8, intent(in), dimension(n_pc) :: pc_in
        real*8, intent(out), dimension(n_pc) :: m_out, r_out, lam_out
        
        integer :: i
        
        ! Store EOS in module variables
        n_tab = n_eos
        if (allocated(eos_p)) deallocate(eos_p)
        if (allocated(eos_e)) deallocate(eos_e)
        allocate(eos_p(n_tab), eos_e(n_tab))
        
        eos_p = p_in
        eos_e = e_in
        
        ! Loop over central pressures and solve
        do i = 1, n_pc
            call solve_single_star(pc_in(i), m_out(i), r_out(i), lam_out(i))
        end do
        
        deallocate(eos_p, eos_e)
    end subroutine tov_mrl_vector

    ! --- RK4 Integration for a Single Star ---
    subroutine solve_single_star(pc, m_final, r_final, lam_final)
        real*8, intent(in) :: pc
        real*8, intent(out) :: m_final, r_final, lam_final
        
        integer, parameter :: max_steps = 100000
        real*8 :: p_prf, e_prf, em_prf, y_prf, rad, h
        real*8 :: k1, k2, k3, k4, l1, l2, l3, l4, m1, m2, m3, m4
        real*8 :: pfunc1, pfunc2, pfunc3, pfunc4
        real*8 :: emfunc1, emfunc2, emfunc3, emfunc4
        real*8 :: yfunc1, yfunc2, yfunc3, yfunc4
        real*8 :: p_lim, c, ak2, td, y_surf
        integer :: i
        
        p_lim = eos_p(1) * 1.01d0 
        rad = 1.0d-6
        p_prf = pc
        e_prf = get_eng(pc)
        em_prf = 0.0d0
        y_prf = 2.0d0
        
        h = 0.0001d0
        
        do i = 1, max_steps
            if (p_prf <= p_lim) exit
            
            if (i > 1) then
                call get_derivatives(rad, p_prf, em_prf, y_prf, pfunc1, emfunc1, yfunc1)
                if (abs(emfunc1/em_prf - pfunc1/p_prf) > 1.0d-14) then
                    h = 0.01d0 / abs(emfunc1/em_prf - pfunc1/p_prf)
                end if
            end if
            
            call get_derivatives(rad, p_prf, em_prf, y_prf, pfunc1, emfunc1, yfunc1)
            k1 = h * pfunc1; l1 = h * emfunc1; m1 = h * yfunc1
            
            call get_derivatives(rad + h/2.0d0, p_prf + k1/2.0d0, em_prf + l1/2.0d0, y_prf + m1/2.0d0, pfunc2, emfunc2, yfunc2)
            k2 = h * pfunc2; l2 = h * emfunc2; m2 = h * yfunc2
            
            call get_derivatives(rad + h/2.0d0, p_prf + k2/2.0d0, em_prf + l2/2.0d0, y_prf + m2/2.0d0, pfunc3, emfunc3, yfunc3)
            k3 = h * pfunc3; l3 = h * emfunc3; m3 = h * yfunc3
            
            call get_derivatives(rad + h, p_prf + k3, em_prf + l3, y_prf + m3, pfunc4, emfunc4, yfunc4)
            k4 = h * pfunc4; l4 = h * emfunc4; m4 = h * yfunc4
            
            p_prf = p_prf + (k1 + 2.0d0*k2 + 2.0d0*k3 + k4) / 6.0d0
            em_prf = em_prf + (l1 + 2.0d0*l2 + 2.0d0*l3 + l4) / 6.0d0
            y_prf = y_prf + (m1 + 2.0d0*m2 + 2.0d0*m3 + m4) / 6.0d0
            rad = rad + h
        end do
        
        m_final = em_prf / xmsun
        r_final = rad
        
        c = em_prf / rad
        e_prf = get_eng(p_prf)
        
        y_surf = y_prf - 4.0d0 * pi * (rad**3.0d0) * e_prf / em_prf
        
        ak2 = 8.0d0*(c**5.0d0)*((1.0d0-2.0d0*c)**2.0d0)*(2.0d0+2.0d0*c*(y_surf-1.0d0)-y_surf) / &
              (5.0d0*(2.0d0*c*(6.0d0-3.0d0*y_surf+3.0d0*c*(5.0d0*y_surf-8.0d0)) + &
              4.0d0*(c**3.0d0)*(13.0d0-11.0d0*y_surf+c*(3.0d0*y_surf-2.0d0)+2.0d0*(c**2.0d0)*(1.0d0+y_surf)) + &
              3.0d0*((1.0d0-2.0d0*c)**2.0d0)*(2.0d0-y_surf+2.0d0*c*(y_surf-1.0d0))*log(1.0d0-2.0d0*c)))
              
        td = (2.0d0/3.0d0) * ak2 * (rad**5.0d0)
        lam_final = td / (em_prf**5.0d0)
        
    end subroutine solve_single_star

    ! --- Structure Equations ---
    subroutine get_derivatives(xr, xp, xm, xy, pfc, emfc, yfc)
        real*8, intent(in) :: xr, xp, xm, xy
        real*8, intent(out) :: pfc, emfc, yfc
        real*8 :: xe, cs, vol, ycoef1, ycoef2, yterm1, yterm2, yterm3
        
        xe = get_eng(xp)
        cs = get_vs(xp)
        vol = 4.0d0 * pi * xr * xr * xr
        
        pfc = -(xe + xp) * (xm + vol * xp) / (xr * (xr - 2.0d0 * xm))
        emfc = 4.0d0 * pi * xr * xr * xe
        
        ycoef1 = -1.0d0 / xr
        ycoef2 = -(xr + vol * (xp - xe)) / (xr * (xr - 2.0d0 * xm))
        yterm1 = (4.0d0 * (xm + vol * xp)**2.0d0) / (xr * (xr - 2.0d0 * xm)**2.0d0)
        yterm2 = 6.0d0 / (xr - 2.0d0 * xm)
        yterm3 = -4.0d0 * pi * xr * xr * (5.0d0 * xe + 9.0d0 * xp + (xe + xp)/cs) / (xr - 2.0d0 * xm)
        
        yfc = ycoef1 * xy * xy + ycoef2 * xy + yterm1 + yterm2 + yterm3
    end subroutine get_derivatives

    ! --- Fast Binary Search Interpolation for Energy ---
    real*8 function get_eng(ap)
        real*8, intent(in) :: ap
        integer :: low, high, mid
        
        if (ap <= eos_p(1)) then
            get_eng = eos_e(1)
            return
        else if (ap >= eos_p(n_tab)) then
            get_eng = eos_e(n_tab)
            return
        end if
        
        low = 1
        high = n_tab
        do while (low <= high)
            mid = (low + high) / 2
            if (eos_p(mid) < ap) then
                low = mid + 1
            else
                high = mid - 1
            end if
        end do
        
        get_eng = eos_e(high) * exp( log(ap/eos_p(high)) * &
                  log(eos_e(high+1)/eos_e(high)) / log(eos_p(high+1)/eos_p(high)) )
    end function get_eng

    ! --- Fast Binary Search Interpolation for Sound Speed Squared ---
    real*8 function get_vs(ap)
        real*8, intent(in) :: ap
        integer :: low, high, mid
        
        if (ap <= eos_p(1)) then
            get_vs = (eos_p(2) - eos_p(1)) / (eos_e(2) - eos_e(1))
            return
        else if (ap >= eos_p(n_tab)) then
            get_vs = (eos_p(n_tab) - eos_p(n_tab-1)) / (eos_e(n_tab) - eos_e(n_tab-1))
            return
        end if
        
        low = 1
        high = n_tab
        do while (low <= high)
            mid = (low + high) / 2
            if (eos_p(mid) < ap) then
                low = mid + 1
            else
                high = mid - 1
            end if
        end do
        
        get_vs = (eos_p(high+1) - eos_p(high)) / (eos_e(high+1) - eos_e(high))
    end function get_vs

end module tov_module