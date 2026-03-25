SUBROUTINE tov_mrl_vector(e_eos, p_eos, pcs, m_out, r_out, l_out, n_eos, n_pc)
    implicit none
    integer, intent(in) :: n_eos, n_pc
    real*8, intent(in) :: e_eos(n_eos), p_eos(n_eos), pcs(n_pc)
    real*8, intent(out) :: m_out(n_pc), r_out(n_pc), l_out(n_pc)

    integer :: i, j
    real*8 :: pc, r_curr, p_curr, m_curr, y_curr, h, pi, xmsun, p_lim
    real*8 :: k1, k2, k3, k4, l1, l2, l3, l4, m1, m2, m3, m4
    real*8 :: r_prev, p_prev, m_prev, frac, r_surf, m_surf, y_surf
    real*8 :: ak2, c, xe

    !f2py intent(in) e_eos, p_eos, pcs, n_eos, n_pc
    !f2py intent(out) m_out, r_out, l_out

    pi = 4.0d0 * datan(1.0d0)
    xmsun = 1.4766d0
    p_lim = 5.0d-18
    h = 0.01d0

    do j = 1, n_pc

        pc = pcs(j)

        ! --- Guard: pc must be inside EOS ---
        if (pc .gt. p_eos(n_eos) .or. pc .lt. p_eos(1)) then
            m_out(j) = 0.0d0
            r_out(j) = 0.0d0
            l_out(j) = 0.0d0
            cycle
        end if

        r_curr = 1.0d-6
        p_curr = pc
        xe = get_eng(pc)
        m_curr = (4.0d0/3.0d0) * pi * r_curr**3 * xe
        y_curr = 2.0d0

        do i = 1, 100000

            p_prev = p_curr
            r_prev = r_curr
            m_prev = m_curr

            call derivatives(r_curr, p_curr, m_curr, y_curr, k1, l1, m1)
            call derivatives(r_curr+h/2, p_curr+k1*h/2, m_curr+l1*h/2, y_curr+m1*h/2, k2, l2, m2)
            call derivatives(r_curr+h/2, p_curr+k2*h/2, m_curr+l2*h/2, y_curr+m2*h/2, k3, l3, m3)
            call derivatives(r_curr+h, p_curr+k3*h, m_curr+l3*h, y_curr+m3*h, k4, l4, m4)

            p_curr = p_curr + h*(k1 + 2.d0*k2 + 2.d0*k3 + k4)/6.d0
            m_curr = m_curr + h*(l1 + 2.d0*l2 + 2.d0*l3 + l4)/6.d0
            y_curr = y_curr + h*(m1 + 2.d0*m2 + 2.d0*m3 + m4)/6.d0
            r_curr = r_curr + h

            if (p_curr .le. p_lim) then

                ! --- Linear surface interpolation ---
                frac = p_prev / (p_prev - p_curr)
                r_surf = r_prev + frac * h
                m_surf = m_prev + frac * (m_curr - m_prev)
                y_surf = y_curr

                ! --- Surface correction (IMPORTANT) ---
                xe = get_eng(p_prev)
                if (m_surf .gt. 0.d0) then
                    y_surf = y_surf - 4.d0*pi*(r_surf**3)*xe/m_surf
                end if

                r_out(j) = r_surf
                m_out(j) = m_surf / xmsun

                ! --- Love number ---
                c = m_surf / r_surf
                if (c .gt. 0.d0 .and. c .lt. 0.5d0) then
                    ak2 = calc_love(c, y_surf)
                    l_out(j) = (2.d0/3.d0) * ak2 / (c**5)
                else
                    l_out(j) = 0.d0
                end if

                exit
            end if

        end do

    end do

CONTAINS

    ! --- LOG INTERPOLATION (like original code) ---
    function get_eng(ap) result(eng)
        real*8, intent(in) :: ap
        real*8 :: eng
        integer :: k
        real*8 :: xp1, xp2, xe1, xe2

        do k = 1, n_eos-1
            xp1 = p_eos(k)
            xp2 = p_eos(k+1)
            xe1 = e_eos(k)
            xe2 = e_eos(k+1)

            if (ap >= xp1 .and. ap <= xp2) then
                eng = xe1 * dexp(dlog(ap/xp1) * dlog(xe2/xe1) / dlog(xp2/xp1))
                return
            end if
        end do

        eng = e_eos(1)
    end function


    subroutine derivatives(xr, xp, xm, xy, dfp, dfm, dfy)
        real*8, intent(in) :: xr, xp, xm, xy
        real*8, intent(out) :: dfp, dfm, dfy
        real*8 :: xe, vol, cs2
        integer :: k

        xe = get_eng(xp)
        vol = 4.d0*pi*xr**3

        ! --- sound speed ---
        cs2 = 1.d-6
        do k = 1, n_eos-1
            if (xp >= p_eos(k) .and. xp <= p_eos(k+1)) then
                cs2 = (p_eos(k+1)-p_eos(k)) / (e_eos(k+1)-e_eos(k))
                exit
            end if
        end do
        if (cs2 .le. 1.d-10) cs2 = 1.d-10

        dfp = -(xe+xp)*(xm+vol*xp)/(xr*(xr-2.d0*xm))
        dfm = 4.d0*pi*xr**2*xe

        dfy = -(xy**2/xr) - xy*(xr+vol*(xp-xe))/(xr*(xr-2.d0*xm)) + &
              (4.d0*(xm+vol*xp)**2)/(xr*(xr-2.d0*xm)**2) + &
              6.d0/(xr-2.d0*xm) - &
              4.d0*pi*xr**2*(5.d0*xe + 9.d0*xp + (xe+xp)/cs2)/(xr-2.d0*xm)
    end subroutine


    function calc_love(c, y) result(ak2)
        real*8, intent(in) :: c, y
        real*8 :: ak2

        ak2 = (8.d0/5.d0)*(c**5)*(1.d0-2.d0*c)**2*(2.d0+2.d0*c*(y-1.d0)-y) / &
              (2.d0*c*(6.d0-3.d0*y+3.d0*c*(5.d0*y-8.d0)) + &
               4.d0*c**3*(13.d0-11.d0*y+c*(3.d0*y-2.d0)+2.d0*c**2*(1.d0+y)) + &
               3.d0*(1.d0-2.d0*c)**2*(2.d0-y+2.d0*c*(y-1.d0))*dlog(1.d0-2.d0*c))
    end function

END SUBROUTINE tov_mrl_vector