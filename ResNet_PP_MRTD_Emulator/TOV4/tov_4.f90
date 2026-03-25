module tov_module
    implicit none
    private
    public :: tov_mrl_vector

    real(8), parameter :: pi = 3.141592653589793d0
    real(8), parameter :: P_lim = 1d-10
    real(8), parameter :: safety = 0.3d0

contains

!=========================================================
! VECTOR DRIVER (all stars)
!=========================================================
subroutine tov_mrl_vector(eps_tab, p_tab, pc_array, n_eos, nstars, m_out, r_out, lambda_out)
    implicit none
    integer, intent(in) :: n_eos, nstars
    real(8), intent(in) :: eps_tab(n_eos), p_tab(n_eos)
    real(8), intent(in) :: pc_array(nstars)
    real(8), intent(out) :: m_out(nstars), r_out(nstars), lambda_out(nstars)
    integer :: i

    ! --- f2py directives ---
    !f2py intent(in) eps_tab, p_tab, pc_array, n_eos, nstars
    !f2py intent(out) m_out, r_out, lambda_out

    do i = 1, nstars
        call solve_star(eps_tab, p_tab, n_eos, pc_array(i), &
                        m_out(i), r_out(i), lambda_out(i))
    end do
end subroutine tov_mrl_vector

!=========================================================
! SINGLE STAR SOLVER
!=========================================================
subroutine solve_star(eps_tab, p_tab, n_eos, pc, m_fin, r_fin, lambda_fin)
    implicit none
    integer, intent(in) :: n_eos
    real(8), intent(in) :: eps_tab(n_eos), p_tab(n_eos)
    real(8), intent(in) :: pc
    real(8), intent(out) :: m_fin, r_fin, lambda_fin
    real(8) :: r, m, p, y, dr, dmdr, dpdr, dydr, c, k2, e

    r = 1d-5
    p = pc
    m = 0d0
    y = 2d0

    do while (p > P_lim)
        call interp_eps(p, e, p_tab, eps_tab, n_eos)
        dmdr = 4d0*pi*r**2 * e
        dpdr = - (e + p) * (m + 4d0*pi*r**3*p) / (r * (r - 2d0*m))
        dydr = - (y*y + y*F(r,m,p,e) + r*r*Q(r,m,p,e)) / r

        ! Adaptive step size
        if (abs(dpdr) > 0d0) then
            dr = safety * abs(p / dpdr)
        else
            dr = 1d-3
        end if
        dr = min(dr, 0.5d0)
        dr = max(dr, 1d-6)

        ! RK4 step
        call rk4_step(r, m, p, y, dr, eps_tab, p_tab, n_eos)
        r = r + dr
    end do

    ! Final outputs
    m_fin = m
    r_fin = r
    c = m / r
    k2 = compute_k2(c, y)
    lambda_fin = (2d0/3d0) * k2 / (c**5)
end subroutine solve_star

!=========================================================
! RK4 STEP
!=========================================================
subroutine rk4_step(r, m, p, y, dr, eps_tab, p_tab, n_eos)
    implicit none
    integer, intent(in) :: n_eos
    real(8), intent(in) :: eps_tab(n_eos), p_tab(n_eos), dr
    real(8), intent(inout) :: r, m, p, y
    real(8) :: k1m, k2m, k3m, k4m
    real(8) :: k1p, k2p, k3p, k4p
    real(8) :: k1y, k2y, k3y, k4y
    real(8) :: e

    call interp_eps(p, e, p_tab, eps_tab, n_eos)
    call derivs(r, m, p, y, e, k1m, k1p, k1y)
    call interp_eps(p + 0.5d0*dr*k1p, e, p_tab, eps_tab, n_eos)
    call derivs(r + 0.5d0*dr, m + 0.5d0*dr*k1m, p + 0.5d0*dr*k1p, y + 0.5d0*dr*k1y, e, k2m, k2p, k2y)
    call interp_eps(p + 0.5d0*dr*k2p, e, p_tab, eps_tab, n_eos)
    call derivs(r + 0.5d0*dr, m + 0.5d0*dr*k2m, p + 0.5d0*dr*k2p, y + 0.5d0*dr*k2y, e, k3m, k3p, k3y)
    call interp_eps(p + dr*k3p, e, p_tab, eps_tab, n_eos)
    call derivs(r + dr, m + dr*k3m, p + dr*k3p, y + dr*k3y, e, k4m, k4p, k4y)

    m = m + dr*(k1m + 2d0*k2m + 2d0*k3m + k4m)/6d0
    p = p + dr*(k1p + 2d0*k2p + 2d0*k3p + k4p)/6d0
    y = y + dr*(k1y + 2d0*k2y + 2d0*k3y + k4y)/6d0
end subroutine rk4_step

!=========================================================
! DERIVATIVES
!=========================================================
subroutine derivs(r, m, p, y, e, dmdr, dpdr, dydr)
    implicit none
    real(8), intent(in) :: r, m, p, y, e
    real(8), intent(out) :: dmdr, dpdr, dydr

    dmdr = 4d0*pi*r**2 * e
    dpdr = - (e + p) * (m + 4d0*pi*r**3*p) / (r * (r - 2d0*m))
    dydr = - (y*y + y*F(r,m,p,e) + r*r*Q(r,m,p,e)) / r
end subroutine derivs

!=========================================================
! AUXILIARY FUNCTIONS
!=========================================================
real(8) function F(r,m,p,e)
    implicit none
    real(8), intent(in) :: r,m,p,e
    F = (1d0 - 4d0*pi*r*r*(e - p)) / (1d0 - 2d0*m/r)
end function

real(8) function Q(r,m,p,e)
    implicit none
    real(8), intent(in) :: r,m,p,e
    Q = (4d0*pi*(5d0*e + 9d0*p + (e+p))) / (1d0 - 2d0*m/r)
end function

real(8) function compute_k2(c, y)
    implicit none
    real(8), intent(in) :: c, y
    compute_k2 = (8d0/5d0)*c**5*(1d0-2d0*c)**2*(2d0+2d0*c*(y-1d0)-y) / &
                 (2d0*c*(6d0-3d0*y+3d0*c*(5d0*y-8d0)) + &
                  4d0*c**3*(13d0-11d0*y) + &
                  8d0*c**5*(1d0+y) + &
                  3d0*(1d0-2d0*c)**2*(2d0-y+2d0*c*(y-1d0))*log(1d0-2d0*c))
end function compute_k2

!=========================================================
! EOS STUBS
!=========================================================
subroutine interp_eps(p, eps_out, p_tab, eps_tab, n_eos)
    implicit none
    integer, intent(in) :: n_eos
    real(8), intent(in) :: p
    real(8), intent(in) :: p_tab(n_eos), eps_tab(n_eos)
    real(8), intent(out) :: eps_out
    integer :: i
    real(8) :: logp, logp1, logp2, loge1, loge2

    ! Edge cases
    if (p <= p_tab(1)) then
        eps_out = eps_tab(1)
        return
    else if (p >= p_tab(n_eos)) then
        eps_out = eps_tab(n_eos)
        return
    end if

    logp = log(p)

    ! Linear search + log-log interpolation
    do i = 2, n_eos
        if (p <= p_tab(i) .and. p >= p_tab(i-1)) then
            logp1 = log(p_tab(i-1))
            logp2 = log(p_tab(i))
            loge1 = log(eps_tab(i-1))
            loge2 = log(eps_tab(i))
            eps_out = exp(loge1 + (logp - logp1) * (loge2 - loge1) / (logp2 - logp1))
            return
        end if
    end do
end subroutine interp_eps

end module tov_module