! Copyright 2019 Xanadu Quantum Technologies Inc.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.



module torontonian
    use kinds
    use vars
    use omp_lib
    use ISO_FORTRAN_ENV
    use linpack_q_complex

    implicit none

    contains

    subroutine hello(varin, varout)
      use kinds
   
      real(dp), intent(in) :: varin
      real(wp) :: tmp
      real(dp), intent(out) :: varout

      !f2py intent(in) :: varin
      !f2py intent(out) :: varout

      print*, 'varin=', varin
      tmp = real(varin, wp)
      varout = tmp!real(tmp, dp)
      print*, 'varout=', varout
 
      print*, "hello from Toronotnian!"
    end subroutine hello

    !function tor(mat)
    subroutine tor(matin, tor_out)
      use kinds
      use vars
      use omp_lib
      use ISO_FORTRAN_ENV

      implicit none

        complex(dp), intent(in) :: matin(:, :)
        complex(wp), allocatable :: mat(:, :)
        complex(dp), allocatable :: mat_dp1(:, :), mat_dp2(:, :), mat_dp3(:, :)
        !complex(wp) :: tor
        complex(dp), intent(out) :: tor_out
        real(dp) :: tmp, tmp2, tmp3
        real(wp) :: tmpqp

        ! local variables
        integer(ip) :: n, i, j, k, total, cntr, ell, ii, nthreads
        complex(wp) :: tmpsum_complex, invdet_complex

        ! result variables
        complex(wp), allocatable :: submat_comp(:, :)

        ! determinant work variables
        complex(wp)                  :: det_complex(2)
        integer(kind=4)              :: info, lda, nn, job
        integer(kind=4), allocatable :: ipvt(:)

        real(wp), allocatable    :: work(:)
        complex(wp), allocatable :: work_complex(:)
        integer(ip), allocatable :: bin(:), iter(:), iter2(:)


        !f2py intent(in) :: matin
        !f2py intent(out) :: tor_out

        n = size(matin(1,:))

        allocate(mat(1:n, 1:n), mat_dp1(1:n, 1:n),mat_dp2(1:n, 1:n), mat_dp3(1:n, 1:n))

        mat_dp1(:, :) = (0.0_dp, 0.0_dp)

        print*, real(mat_dp1(1,1), wp)
        print*, real(matin(1,1), wp) - real(real(matin(1,1)), dp)

        forall (i=1:n, j=1:n) mat(i,j) = zone*real(matin(i,j), wp) + zi*real(aimag(matin(i,j)), wp)!matdp(i,j)!(real(matdp(i,j), wp), aimag(matdp(i,j), wp))

        forall (i=1:n, j=1:n) mat_dp1(i, j) = (1.0_dp, 0.0_dp)*real(matin(i,j), dp) + (0.0_dp, 1.0_dp)*real(aimag(matin(i,j)), dp)

        tmp = real(mat(1,1), dp)
        tmp2 = mat(1,1) - real(tmp, wp)
        tmp3 = (mat(1,1) - real(tmp, wp)) - real(tmp2, wp)
        tmpqp = real(tmp, wp) + real(tmp2, wp) + real(tmp3, wp)

        print*, tmp, tmp2, tmp3
        print*, tmpqp 

        ell = n/2

        allocate(bin(1:ell)) !, set(1:j))

        tmpsum_complex = zzero

        !$OMP PARALLEL DO shared(mat, ell, mattype) private(ii, bin, total, submat_comp, &
        !$OMP                    cntr, k, iter, iter2, ipvt, nn, work_complex, det_complex, job, &
        !$OMP                     invdet_complex, lda) reduction(+:tmpsum_complex)

        do ii=0, 2**ell-1
            call dec2bin(ii, ell, bin, total)

            allocate(iter(1:total), iter2(1:2*total), submat_comp(1:2*total, 1:2*total))

            cntr = 1
            do k=1,ell
                if (bin(k) == 1) then
                    iter(cntr) = k
                    cntr = cntr + 1
                end if
            end do

            iter2(1:total) = iter(1:total)
            iter2(total+1:2*total) = iter(1:total) + ell

            forall (i=1:2*total,j=1:2*total) submat_comp(i, j) = -mat(iter2(i), iter2(j))
            forall (i=1:2*total) submat_comp(i, i) = zone + submat_comp(i, i)

            lda = 2*total
            nn = 2*total

            allocate(ipvt(1:nn), work_complex(1:nn))

            call qgefa_complex(submat_comp, lda, nn, ipvt, info)
            job = 10
            call qgedi_complex (submat_comp, lda, nn, ipvt, det_complex, work_complex, job )
            invdet_complex = zone/det_complex(1) * 10.0_wp**(-real(det_complex(2)))
            tmpsum_complex = tmpsum_complex + (-1.0_wp)**(ell-total)*sqrt(invdet_complex)
            !print*, tmpsum_complex
            deallocate(iter, iter2, submat_comp, ipvt, work_complex)
        end do
        !$OMP END PARALLEL DO

        !tor = (1.0_wp, 0.0_wp)!tmpsum_complex
        tor_out = tmpsum_complex

    end subroutine tor
    !end function tor

    subroutine dec2bin (kk, nnn, matt, summ)
        integer(ip), intent(in)  :: kk, nnn
        integer(ip), intent(out) :: matt(1:nnn)
        integer(ip), intent(out) :: summ

        ! local variables
        integer(ip) :: i, k

        matt(:) = 0

        k = kk
        i = nnn

        do while (k > 0 .and. i>0)
            matt(i) = mod(k, 2)
            k = k/2 !floor(real(k, wp)/2.0_wp)
            i = i-1
        end do

        summ = sum(matt(1:nnn))

    end subroutine dec2bin



end module torontonian
