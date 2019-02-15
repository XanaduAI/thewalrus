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

! Module containing custom precision types
module kinds
    implicit none

    integer, parameter :: sp = selected_real_kind(5,30)
    integer, parameter :: dp = selected_real_kind(9,99)
    integer, parameter :: qp = selected_real_kind(20, 199)
    integer, parameter :: wp = qp
    integer, parameter :: ip = selected_int_kind(16)

    ! These empty arrays are used to initialize variables to either the min or
    ! max possible number of kind wp or integer.
    real(wp), dimension(2:1)    :: empty
    integer(ip), dimension(2:1) :: iempty
end module kinds


! Module containing some common variables
module vars
    use kinds
    implicit none

    ! real(wp), dimension(:, :), allocatable    :: mat, inv_mat
    real(wp), dimension(:, :), allocatable    :: mat_comp_real, mat_comp_imag
    complex(wp), dimension(:, :), allocatable :: inv_mat_comp


    complex(wp), parameter :: zzero = (0.0_wp, 0.0_wp), zone = (1.0_wp, 0.0_wp), zi = (0.0_wp, 1.0_wp)

    character :: mattype*10

end module vars


module torontonian
    use kinds
    use vars
    use omp_lib
    use linpack_q_complex

    implicit none

    contains

    function tor(mat)
        complex(wp), intent(in) :: mat(:, :)
        complex(wp) :: tor

        ! local variables
        integer(ip) :: n, i, j, k, total, cntr, ell, ii
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


        !f2py intent(in) :: mat

        n = size(mat(1,:))

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
            deallocate(iter, iter2, submat_comp, ipvt, work_complex)
        end do
        !$OMP END PARALLEL DO

        tor = tmpsum_complex

    end function tor

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
