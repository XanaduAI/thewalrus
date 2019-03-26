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

    subroutine tor(matin, tor_out)
        complex(dp), intent(in)  :: matin(:, :)
        complex(dp), intent(out) :: tor_out

        ! local variables
        complex(wp), allocatable :: mat(:, :)
        complex(dp), allocatable :: mat_dp1(:, :), mat_dp2(:, :), mat_dp3(:, :)

        real(dp)     :: tmp, tmp2, tmp3
        real(wp)     :: tmpqp
        integer(ip2) :: n, i, j, k, total, cntr, ell, ii, nthreads
        complex(wp)  :: tmpsum_complex, invdet_complex

        ! result variables
        complex(wp), allocatable :: submat_comp(:, :)

        ! determinant work variables
        complex(wp)                  :: det_complex(2)
        integer(kind=4)              :: info, lda, nn, job

        integer(kind=4), allocatable :: ip2vt(:)
        real(wp), allocatable        :: work(:)
        complex(wp), allocatable     :: work_complex(:)
        integer(ip2), allocatable    :: bin(:), iter(:), iter2(:)
        write(*,*)"line 50"
        n = size(matin(1,:))

        allocate(mat(1:n, 1:n))

        forall (i=1:n, j=1:n) mat(i,j) = zone*real(matin(i,j), wp) + zi*real(aimag(matin(i,j)), wp)!matdp(i,j)!(real(matdp(i,j), wp), aimag(matdp(i,j), wp))

        write(*,*)"line 57"
        ell = n/2

        allocate(bin(1:ell)) !, set(1:j))
        write(*,*)"line 61"
        tmpsum_complex = zzero

        !$OMP PARALLEL DO shared(mat, ell, mattype) private(ii, bin, total, submat_comp, &
        !$OMP                    cntr, k, iter, iter2, ip2vt, nn, work_complex, det_complex, job, &
        !$OMP                     invdet_complex, lda) reduction(+:tmpsum_complex)
        do ii=0, 2**ell-1
            write(*,*)"line 68", ii, ell, total
            call dec2bin(ii, ell, bin, total)
            write(*,*)"line 70"
            allocate(iter(1:total), iter2(1:2*total), submat_comp(1:2*total, 1:2*total))
            write(*,*)"line 72"
            cntr = 1
            do k=1,ell
                if (bin(k) == 1) then
                    iter(cntr) = k
                    cntr = cntr + 1
                end if
            end do
            write(*,*)"line 80"
            iter2(1:total) = iter(1:total)
            iter2(total+1:2*total) = iter(1:total) + ell
            write(*,*)"line 83"
            forall (i=1:2*total,j=1:2*total) submat_comp(i, j) = -mat(iter2(i), iter2(j))
            forall (i=1:2*total) submat_comp(i, i) = zone + submat_comp(i, i)
            write(*,*)"line 86"
            lda = 2*total
            nn = 2*total
            write(*,*)"line 89"
            allocate(ip2vt(1:nn), work_complex(1:nn))
            write(*,*)"line 91"
            call qgefa_complex(submat_comp, lda, nn, ip2vt, info)
            job = 10
            write(*,*)"line 94"
            call qgedi_complex(submat_comp, lda, nn, ip2vt, det_complex, work_complex, job )
            write(*,*)"line 96"
            invdet_complex = zone/det_complex(1) * 10.0_wp**(-real(det_complex(2)))
            write(*,*)"line 98"
            tmpsum_complex = tmpsum_complex + (-1.0_wp)**(ell-total)*sqrt(invdet_complex)
            write(*,*)"line 100"
            !print*, tmpsum_complex
            deallocate(iter, iter2, submat_comp, ip2vt, work_complex)
        end do
        !$OMP END PARALLEL DO
        write(*,*)"line 105"
        !tor = (1.0_wp, 0.0_wp)!tmpsum_complex
        tor_out = tmpsum_complex

    end subroutine tor

    subroutine det_real(matin, det_out)
        real(dp), intent(in)  :: matin(:, :)
        real(dp), intent(out) :: det_out

        ! determinant work variables
        real(wp)        :: det(2), tmpdet
        integer(kind=4) :: info, lda, nn, job, i, j
        integer(ip2)    :: n

        integer(kind=4), allocatable :: ip2vt(:)
        real(wp), allocatable        :: work(:), mat(:, :)

        nn = size(matin(1,:))
        lda = nn
        write(*,*)"det_real line 125"
        allocate(ip2vt(1:nn), work(1:nn), mat(1:nn,1:nn))
        write(*,*)"det_real line 127"
        forall (i=1:nn,j=1:nn) mat(i,j) = real(matin(i,j), wp)
        write(*,*)"det_real line 129"
        call qgefa(mat, lda, nn, ip2vt, info)
        write(*,*)"det_real line 131"
        job = 10
        write(*,*)"det_real line 133"
        call qgedi(mat, lda, nn, ip2vt, det, work, job )
        write(*,*)"det_real line 135"
        tmpdet = det(1) * 10.0_wp**(real(det(2), wp))

        det_out = real(tmpdet, dp)

        deallocate(ip2vt, work, mat)
    end subroutine det_real

    subroutine det_complex(matin, det_out)
        complex(dp), intent(in)  :: matin(:, :)
        complex(dp), intent(out) :: det_out

        ! determinant work variables
        complex(wp)     :: det(2), tmpdet
        integer(kind=4) :: info, lda, nn, job, i, j
        integer(ip2)    :: n

        integer(kind=4), allocatable :: ip2vt(:)
        complex(wp), allocatable     :: work(:), mat(:, :)

        nn = size(matin(1,:))
        lda = nn
        write(*,*)"det_complex line 157"
        allocate(ip2vt(1:nn), work(1:nn), mat(1:nn,1:nn))
        write(*,*)"det_complex line 159"
        forall (i=1:nn, j=1:nn) mat(i,j) = zone*real(matin(i,j), wp) + zi*real(aimag(matin(i,j)), wp)
        write(*,*)"det_complex line 161"
        call qgefa_complex(mat, lda, nn, ip2vt, info)
        write(*,*)"det_complex line 163"
        job = 10
        write(*,*)"det_complex line 165"
        call qgedi_complex(mat, lda, nn, ip2vt, det, work, job)
        write(*,*)"det_complex line 167"
        tmpdet = det(1) * 10.0_wp**(real(det(2), wp))
        write(*,*)"det_complex line 169"
        det_out = zone*real(tmpdet, wp) + zi*real(aimag(tmpdet), wp)
        write(*,*)"det_complex line 171"
        deallocate(ip2vt, work, mat)
    end subroutine det_complex

    subroutine dec2bin (kk, nnn, matt, summ)
        integer(ip2), intent(in)  :: kk, nnn
        integer(ip2), intent(out) :: matt(1:nnn)
        integer(ip2), intent(out) :: summ

        ! local variables
        integer(ip2) :: i, k
        write(*,*)"dec2bin line 182"
        matt(:) = 0

        k = kk
        i = nnn

        do while (k > 0 .and. i>0)
            matt(i) = mod(k, 2)
            k = k/2 !floor(real(k, wp)/2.0_wp)
            i = i-1
        end do
        write(*,*)"dec2bin line 193"
        summ = sum(matt(1:nnn))
    end subroutine dec2bin

end module torontonian
