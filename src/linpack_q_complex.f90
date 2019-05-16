module linpack_q_complex
  use kinds

  implicit none

contains

function iqamax ( n, x, incx )

!*****************************************************************************80
!
!! IQAMAX finds the index of the vector element of maximum absolute value.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    08 April 1999
!
!  Author:
!
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
!    Algorithm 539: 
!    Basic Linear Algebra Subprograms for Fortran Usage,
!    ACM Transactions on Mathematical Software,
!    Volume 5, Number 3, September 1979, pages 308-323.
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the number of entries in the vector.
!
!    Input, real ( kind = qp ) X(*), the vector to be examined.
!
!    Input, integer ( kind = 4 ) INCX, the increment between successive entries of SX.
!
!    Output, integer ( kind = 4 ) IQAMAX, the index of the element of SX of maximum
!    absolute value.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) i
  integer ( kind = 4 ) incx
  integer ( kind = 4 ) iqamax
  integer ( kind = 4 ) ix
  integer ( kind = 4 ) n
  real ( kind = wp ) samax
  real ( kind = wp ) x(*)

  if ( n <= 0 ) then

    iqamax = 0

  else if ( n == 1 ) then

    iqamax = 1

  else if ( incx == 1 ) then

    iqamax = 1
    samax = abs ( x(1) )

    do i = 2, n

      if ( samax < abs ( x(i) ) ) then
        iqamax = i
        samax = abs ( x(i) )
      end if

    end do

  else

    if ( 0 <= incx ) then
      ix = 1
    else
      ix = ( - n + 1 ) * incx + 1
    end if

    iqamax = 1
    samax = abs ( x(ix) )

    ix = ix + incx

    do i = 2, n
      if ( samax < abs ( x(ix) ) ) then
        iqamax = i
        samax = abs ( x(ix) )
      end if
      ix = ix + incx
    end do

  end if

  return
end
subroutine qaxpy ( n, sa, x, incx, y, incy )
  use kinds

!*****************************************************************************80
!
!! QAXPY adds a constant times one vector to another.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    08 April 1999
!
!  Author:
!
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
!    Algorithm 539: 
!    Basic Linear Algebra Subprograms for Fortran Usage,
!    ACM Transactions on Mathematical Software,
!    Volume 5, Number 3, September 1979, pages 308-323.
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the number of entries in the vector.
!
!    Input, real SA, the multiplier.
!
!    Input, real ( kind = qp ) X(*), the vector to be scaled and added to Y.
!
!    Input, integer ( kind = 4 ) INCX, the increment between successive entries of X.
!
!    Input/output, real ( kind = qp ) Y(*), the vector to which a 
!    multiple of X is to be added.
!
!    Input, integer ( kind = 4 ) INCY, the increment between successive entries of Y.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) i
  integer ( kind = 4 ) incx
  integer ( kind = 4 ) incy
  integer ( kind = 4 ) ix
  integer ( kind = 4 ) iy
  integer ( kind = 4 ) n
  real ( kind = wp ) sa
  real ( kind = wp ) x(*)
  real ( kind = wp ) y(*)

  if ( n <= 0 ) then

  else if ( sa == 0.0Q+00 ) then

  else if ( incx == 1 .and. incy == 1 ) then

    y(1:n) = y(1:n) + sa * x(1:n)

  else

    if ( 0 <= incx ) then
      ix = 1
    else
      ix = ( - n + 1 ) * incx + 1
    end if

    if ( 0 <= incy ) then
      iy = 1
    else
      iy = ( - n + 1 ) * incy + 1
    end if

    do i = 1, n
      y(iy) = y(iy) + sa * x(ix)
      ix = ix + incx
      iy = iy + incy
    end do

  end if

  return
end
subroutine qgeco ( a, lda, n, ipvt, rcond, z )
  use kinds

!*****************************************************************************80
!
!! QGECO factors a matrix and estimates its condition number.
!
!  Discussion:
!
!    If RCOND is not needed, QGEFA is slightly faster.
!
!    To solve A * X = B, follow QGECO by QGESL.
!
!    To compute inverse ( A ) * C, follow QGECO by QGESL.
!
!    To compute determinant ( A ), follow QGECO by QGEDI.
!
!    To compute inverse ( A ), follow QGECO by QGEDI.
!
!    For the system A * X = B, relative perturbations in A and B
!    of size EPSILON may cause relative perturbations in X of size
!    EPSILON/RCOND.
!
!    If RCOND is so small that the logical expression
!      1.0 + RCOND == 1.0
!    is true, then A may be singular to working precision.  In particular,
!    RCOND is zero if exact singularity is detected or the estimate
!    underflows.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    15 March 2016
!
!  Author:
!
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!  Parameters:
!
!    Input/output, real ( kind = qp ) A(LDA,N).  On input, a matrix to be
!    factored.  On output, the LU factorization of the matrix.
!
!    Input, integer ( kind = 4 ) LDA, the leading dimension of the array A.
!
!    Input, integer ( kind = 4 ) N, the order of the matrix A.
!
!    Output, integer ( kind = 4 ) IPVT(N), the pivot indices.
!
!    Output, real ( kind = qp ) RCOND, an estimate of the reciprocal
!    condition number of A.
!
!    Output, real ( kind = qp ) Z(N), a work vector whose contents are usually
!    unimportant.  If A is close to a singular matrix, then Z is an
!    approximate null vector in the sense that
!      norm ( A * Z ) = RCOND * norm ( A ) * norm ( Z ).
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) lda
  integer ( kind = 4 ) n

  real ( kind = wp ) a(lda,n)
  real ( kind = wp ) anorm
  real ( kind = wp ) ek
  integer ( kind = 4 ) info
  integer ( kind = 4 ) ipvt(n)
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  real ( kind = wp ) rcond
  real ( kind = wp ) s
  real ( kind = wp ) sm
  real ( kind = wp ) t
  real ( kind = wp ) wk
  real ( kind = wp ) wkm
  real ( kind = wp ) ynorm
  real ( kind = wp ) z(n)
!
!  Compute the L1 norm of A.
!
  anorm = 0.0Q+00
  do j = 1, n
    anorm = max ( anorm, sum ( abs ( a(1:n,j) ) ) )
  end do
!
!  Compute the LU factorization.
!
  call qgefa ( a, lda, n, ipvt, info )
!
!  RCOND = 1 / ( norm(A) * (estimate of norm(inverse(A))) )
!
!  estimate of norm(inverse(A)) = norm(Z) / norm(Y)
!
!  where
!    A * Z = Y
!  and
!    A' * Y = E
!
!  The components of E are chosen to cause maximum local growth in the
!  elements of W, where U'*W = E.  The vectors are frequently rescaled
!  to avoid overflow.
!
!  Solve U' * W = E.
!
  ek = 1.0Q+00
  z(1:n) = 0.0Q+00

  do k = 1, n

    if ( z(k) /= 0.0Q+00 ) then
      ek = sign ( ek, - z(k) )
    end if

    if ( abs ( a(k,k) ) < abs ( ek - z(k) ) ) then
      s = abs ( a(k,k) ) / abs ( ek - z(k) )
      z(1:n) = s * z(1:n)
      ek = s * ek
    end if

    wk = ek - z(k)
    wkm = - ek - z(k)
    s = abs ( wk )
    sm = abs ( wkm )

    if ( a(k,k) /= 0.0Q+00 ) then
      wk = wk / a(k,k)
      wkm = wkm / a(k,k)
    else
      wk = 1.0Q+00
      wkm = 1.0Q+00
    end if

    if ( k + 1 <= n ) then

      do j = k + 1, n
        sm = sm + abs ( z(j) + wkm * a(k,j) )
        z(j) = z(j) + wk * a(k,j)
        s = s + abs ( z(j) )
      end do

      if ( s < sm ) then
        t = wkm - wk
        wk = wkm
        z(k+1:n) = z(k+1:n) + t * a(k,k+1:n)
      end if

    end if

    z(k) = wk

  end do

  z(1:n) = z(1:n) / sum ( abs ( z(1:n) ) )
!
!  Solve L' * Y = W
!
  do k = n, 1, -1

    z(k) = z(k) + dot_product ( a(k+1:n,k), z(k+1:n) )

    if ( 1.0Q+00 < abs ( z(k) ) ) then
      z(1:n) = z(1:n) / abs ( z(k) )
    end if

    l = ipvt(k)

    t    = z(l)
    z(l) = z(k)
    z(k) = t

  end do

  z(1:n) = z(1:n) / sum ( abs ( z(1:n) ) )

  ynorm = 1.0Q+00
!
!  Solve L * V = Y.
!
  do k = 1, n

    l = ipvt(k)

    t    = z(l)
    z(l) = z(k)
    z(k) = t

    z(k+1:n) = z(k+1:n) + t * a(k+1:n,k)

    if ( 1.0Q+00 < abs ( z(k) ) ) then
      ynorm = ynorm / abs ( z(k) )
      z(1:n) = z(1:n) / abs ( z(k) )
    end if

  end do

  s = sum ( abs ( z(1:n) ) )
  z(1:n) = z(1:n) / s
  ynorm = ynorm / s
!
!  Solve U * Z = V.
!
  do k = n, 1, -1

    if ( abs ( a(k,k) ) < abs ( z(k) ) ) then
      s = abs ( a(k,k) ) / abs ( z(k) )
      z(1:n) = s * z(1:n)
      ynorm = s * ynorm
    end if

    if ( a(k,k) /= 0.0Q+00 ) then
      z(k) = z(k) / a(k,k)
    else
      z(k) = 1.0Q+00
    end if

    z(1:k-1) = z(1:k-1) - z(k) * a(1:k-1,k)

  end do
!
!  Normalize Z in the L1 norm.
!
  s = 1.0Q+00 / sum ( abs ( z(1:n) ) )
  z(1:n) = s * z(1:n)
  ynorm = s * ynorm

  if ( anorm /= 0.0Q+00 ) then
    rcond = ynorm / anorm
  else
    rcond = 0.0Q+00
  end if

  return
end
subroutine qgedi ( a, lda, n, ipvt, det, work, job )
  use kinds

!*****************************************************************************80
!
!! QGEDI: determinant and inverse of a matrix factored by QGECO or QGEFA.
!
!  Discussion:
!
!    A division by zero will occur if the input factor contains
!    a zero on the diagonal and the inverse is requested.
!    It will not occur if the subroutines are called correctly
!    and if QGECO has set 0.0 < RCOND or QGEFA has set INFO == 0.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    15 March 2016
!
!  Author:
!
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!  Parameters:
!
!    Input/output, real ( kind = qp ) A(LDA,N), on input, the  LU factor
!    information, as output by QGECO or QGEFA.  On output, the inverse
!    matrix, if requested.
!
!    Input, integer ( kind = 4 ) LDA, the leading dimension of the array A.
!
!    Input, integer ( kind = 4 ) N, the order of the matrix A.
!
!    Input, integer ( kind = 4 ) IPVT(N), the pivot vector from QGECO or QGEFA.
!
!    Workspace, real ( kind = qp ) WORK(N).
!
!    Output, real ( kind = qp ) DET(2), the determinant of original matrix if
!    requested.  The determinant = DET(1) * 10.0**DET(2)
!    with  1.0 <= abs ( DET(1) ) < 10.0
!    or DET(1) == 0.0.
!
!    Input, integer ( kind = 4 ) JOB, specifies what is to be computed.
!    11, both determinant and inverse.
!    01, inverse only.
!    10, determinant only.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) lda
  integer ( kind = 4 ) n

  real ( kind = wp ) a(lda,n)
  real ( kind = wp ) det(2)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ipvt(n)
  integer ( kind = 4 ) j
  integer ( kind = 4 ) job
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  real ( kind = wp ) t
  real ( kind = wp ) work(n)
!
!  Compute the determinant.
!
  if ( job / 10 /= 0 ) then

    det(1) = 1.0Q+00
    det(2) = 0.0Q+00

    do i = 1, n

      if ( ipvt(i) /= i ) then
        det(1) = -det(1)
      end if

      det(1) = det(1) * a(i,i)

      if ( det(1) == 0.0Q+00 ) then
        exit
      end if

      do while ( abs ( det(1) ) < 1.0Q+00 )
        det(1) = det(1) * 10.0Q+00
        det(2) = det(2) - 1.0Q+00
      end do

      do while ( 10.0Q+00 <= abs ( det(1) ) )
        det(1) = det(1) / 10.0Q+00
        det(2) = det(2) + 1.0Q+00
      end do

    end do

  end if
!
!  Compute inverse(U).
!
  if ( mod ( job, 10 ) /= 0 ) then

    do k = 1, n

      a(k,k) = 1.0Q+00 / a(k,k)
      t = - a(k,k)
      call qscal ( k-1, t, a(1,k), 1 )

      do j = k + 1, n
        t = a(k,j)
        a(k,j) = 0.0Q+00
        call qaxpy ( k, t, a(1,k), 1, a(1,j), 1 )
      end do

    end do
!
!  Form inverse(U) * inverse(L).
!
    do k = n - 1, 1, -1

      work(k+1:n) = a(k+1:n,k)

      a(k+1:n,k) = 0.0Q+00

      do j = k + 1, n
        t = work(j)
        call qaxpy ( n, t, a(1,j), 1, a(1,k), 1 )
      end do

      l = ipvt(k)
      if ( l /= k ) then
        call qswap ( n, a(1,k), 1, a(1,l), 1 )
      end if

    end do

  end if

  return
end
subroutine qgefa ( a, lda, n, ipvt, info )
  use kinds

!*****************************************************************************80
!
!! QGEFA factors a real matrix.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    07 March 2001
!
!  Author:
!
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!  Parameters:
!
!    Input/output, real ( kind = qp ) A(LDA,N).
!    On intput, the matrix to be factored.
!    On output, an upper triangular matrix and the multipliers used to obtain
!    it.  The factorization can be written A=L*U, where L is a product of
!    permutation and unit lower triangular matrices, and U is upper triangular.
!
!    Input, integer ( kind = 4 ) LDA, the leading dimension of A.
!
!    Input, integer ( kind = 4 ) N, the order of the matrix A.
!
!    Output, integer ( kind = 4 ) IPVT(N), the pivot indices.
!
!    Output, integer ( kind = 4 ) INFO, singularity indicator.
!    0, normal value.
!    K, if U(K,K) == 0.  This is not an error condition for this subroutine,
!    but it does indicate that QGESL or QGEDI will divide by zero if called.
!    Use RCOND in QGECO for a reliable indication of singularity.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) lda
  integer ( kind = 4 ) n

  real ( kind = wp ) a(lda,n)
  integer ( kind = 4 ) info
  integer ( kind = 4 ) ipvt(n)
  !integer ( kind = 4 ) iqamax
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  real ( kind = wp ) t
!
!  Gaussian elimination with partial pivoting.
!
  info = 0

  do k = 1, n - 1
!
!  Find L = pivot index.
!
    l = iqamax ( n - k + 1, a(k,k), 1 ) + k - 1
    ipvt(k) = l
!
!  Zero pivot implies this column already triangularized.
!
    if ( a(l,k) == 0.0Q+00 ) then
      info = k
      cycle
    end if
!
!  Interchange if necessary.
!
    if ( l /= k ) then
      t      = a(l,k)
      a(l,k) = a(k,k)
      a(k,k) = t
    end if
!
!  Compute multipliers.
!
    a(k+1:n,k) = - a(k+1:n,k) / a(k,k)
!
!  Row elimination with column indexing.
!
    do j = k + 1, n
      t = a(l,j)
      if ( l /= k ) then
        a(l,j) = a(k,j)
        a(k,j) = t
      end if
      call qaxpy ( n - k, t, a(k+1,k), 1, a(k+1,j), 1 )
    end do

  end do

  ipvt(n) = n

  if ( a(n,n) == 0.0Q+00 ) then
    info = n
  end if

  return
end
subroutine qgesl ( a, lda, n, ipvt, b, job )
  use kinds

!*****************************************************************************80
!
!! QGESL solves a real general linear system A * X = B.
!
!  Discussion:
!
!    QGESL can solve either of the systems A * X = B or A' * X = B.
!
!    The system matrix must have been factored by QGECO or QGEFA.
!
!    A division by zero will occur if the input factor contains a
!    zero on the diagonal.  Technically this indicates singularity
!    but it is often caused by improper arguments or improper
!    setting of LDA.  It will not occur if the subroutines are
!    called correctly and if QGECO has set 0.0 < RCOND 
!    or QGEFA has set INFO == 0.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    07 March 2001
!
!  Author:
!
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!  Parameters:
!
!    Input, real ( kind = qp ) A(LDA,N), the output from QGECO or QGEFA.
!
!    Input, integer ( kind = 4 ) LDA, the leading dimension of A.
!
!    Input, integer ( kind = 4 ) N, the order of the matrix A.
!
!    Input, integer ( kind = 4 ) IPVT(N), the pivot vector from QGECO or QGEFA.
!
!    Input/output, real ( kind = qp ) B(N).
!    On input, the right hand side vector.
!    On output, the solution vector.
!
!    Input, integer ( kind = 4 ) JOB.
!    0, solve A * X = B;
!    nonzero, solve A' * X = B.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) lda
  integer ( kind = 4 ) n

  real ( kind = wp ) a(lda,n)
  real ( kind = wp ) b(n)
  integer ( kind = 4 ) ipvt(n)
  integer ( kind = 4 ) job
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  real ( kind = wp ) t
!
!  Solve A * X = B.
!
  if ( job == 0 ) then

    do k = 1, n - 1

      l = ipvt(k)
      t = b(l)

      if ( l /= k ) then
        b(l) = b(k)
        b(k) = t
      end if

      call qaxpy ( n - k, t, a(k+1,k), 1, b(k+1), 1 )

    end do

    do k = n, 1, -1
      b(k) = b(k) / a(k,k)
      t = - b(k)
      call qaxpy ( k - 1, t, a(1,k), 1, b(1), 1 )
    end do

  else
!
!  Solve A' * X = B.
!
    do k = 1, n
      t = dot_product ( a(1:k-1,k), b(1:k-1) )
      b(k) = ( b(k) - t ) / a(k,k)
    end do

    do k = n - 1, 1, -1

      b(k) = b(k) + dot_product ( a(k+1:n,k), b(k+1:n) )
      l = ipvt(k)

      if ( l /= k ) then
        t    = b(l)
        b(l) = b(k)
        b(k) = t
      end if

    end do

  end if

  return
end
subroutine qscal ( n, sa, x, incx )
  use kinds

!*****************************************************************************80
!
!! QSCAL scales a vector by a constant.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license. 
!
!  Modified:
!
!    15 March 2016
!
!  Author:
!
!    Original FORTRAN77 version by Charles Lawson, Richard Hanson, 
!    David Kincaid, Fred Krogh.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
!    Algorithm 539, 
!    Basic Linear Algebra Subprograms for Fortran Usage,
!    ACM Transactions on Mathematical Software,
!    Volume 5, Number 3, September 1979, pages 308-323.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the number of entries in the vector.
!
!    Input, real ( kind = qp ) SA, the multiplier.
!
!    Input/output, real ( kind = qp ) X(*), the vector to be scaled.
!
!    Input, integer ( kind = 4 ) INCX, the increment between successive 
!    entries of X.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) i
  integer ( kind = 4 ) incx
  integer ( kind = 4 ) ix
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n
  real ( kind = wp ) sa
  real ( kind = wp ) x(*)

  if ( n <= 0 ) then

  else if ( incx == 1 ) then

    m = mod ( n, 5 )

    x(1:m) = sa * x(1:m)

    do i = m + 1, n, 5
      x(i)   = sa * x(i)
      x(i+1) = sa * x(i+1)
      x(i+2) = sa * x(i+2)
      x(i+3) = sa * x(i+3)
      x(i+4) = sa * x(i+4)
    end do

  else

    if ( 0 <= incx ) then
      ix = 1
    else
      ix = ( - n + 1 ) * incx + 1
    end if

    do i = 1, n
      x(ix) = sa * x(ix)
      ix = ix + incx
    end do

  end if

  return
end
subroutine qswap ( n, x, incx, y, incy )
  use kinds

!*****************************************************************************80
!
!! QSWAP interchanges two vectors.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license. 
!
!  Modified:
!
!    15 March 2016
!
!  Author:
!
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
!    Algorithm 539, 
!    Basic Linear Algebra Subprograms for Fortran Usage,
!    ACM Transactions on Mathematical Software, 
!    Volume 5, Number 3, September 1979, pages 308-323.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the number of entries in the vectors.
!
!    Input/output, real ( kind = qp ) X(*), one of the vectors to swap.
!
!    Input, integer ( kind = 4 ) INCX, the increment between successive 
!    entries of X.
!
!    Input/output, real ( kind = qp ) Y(*), one of the vectors to swap.
!
!    Input, integer ( kind = 4 ) INCY, the increment between successive 
!    elements of Y.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) i
  integer ( kind = 4 ) incx
  integer ( kind = 4 ) incy
  integer ( kind = 4 ) ix
  integer ( kind = 4 ) iy
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n
  real ( kind = wp ) temp
  real ( kind = wp ) x(*)
  real ( kind = wp ) y(*)

  if ( n <= 0 ) then

  else if ( incx == 1 .and. incy == 1 ) then

    m = mod ( n, 3 )

    do i = 1, m
      temp = x(i)
      x(i) = y(i)
      y(i) = temp
    end do

    do i = m + 1, n, 3

      temp = x(i)
      x(i) = y(i)
      y(i) = temp

      temp = x(i+1)
      x(i+1) = y(i+1)
      y(i+1) = temp

      temp = x(i+2)
      x(i+2) = y(i+2)
      y(i+2) = temp

    end do

  else

    if ( 0 <= incx ) then
      ix = 1
    else
      ix = ( - n + 1 ) * incx + 1
    end if

    if ( 0 <= incy ) then
      iy = 1
    else
      iy = ( - n + 1 ) * incy + 1
    end if

    do i = 1, n
      temp = x(ix)
      x(ix) = y(iy)
      y(iy) = temp
      ix = ix + incx
      iy = iy + incy
    end do

  end if

  return
end


function iqamax_complex ( n, x, incx )

!*****************************************************************************80
!
!! IQAMAX finds the index of the vector element of maximum absolute value.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    08 April 1999
!
!  Author:
!
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
!    Algorithm 539: 
!    Basic Linear Algebra Subprograms for Fortran Usage,
!    ACM Transactions on Mathematical Software,
!    Volume 5, Number 3, September 1979, pages 308-323.
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the number of entries in the vector.
!
!    Input, real ( kind = qp ) X(*), the vector to be examined.
!
!    Input, integer ( kind = 4 ) INCX, the increment between successive entries of SX.
!
!    Output, integer ( kind = 4 ) IQAMAX, the index of the element of SX of maximum
!    absolute value.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) i
  integer ( kind = 4 ) incx
  integer ( kind = 4 ) iqamax_complex
  integer ( kind = 4 ) ix
  integer ( kind = 4 ) n
  real ( kind = wp ) samax
  complex ( kind = wp ) x(*)

  if ( n <= 0 ) then

    iqamax_complex = 0

  else if ( n == 1 ) then

    iqamax_complex = 1

  else if ( incx == 1 ) then

    iqamax_complex = 1
    samax = abs ( x(1) )

    do i = 2, n

      if ( samax < abs ( x(i) ) ) then
        iqamax_complex = i
        samax = abs ( x(i) )
      end if

    end do

  else

    if ( 0 <= incx ) then
      ix = 1
    else
      ix = ( - n + 1 ) * incx + 1
    end if

    iqamax_complex = 1
    samax = abs ( x(ix) )

    ix = ix + incx

    do i = 2, n
      if ( samax < abs ( x(ix) ) ) then
        iqamax_complex = i
        samax = abs ( x(ix) )
      end if
      ix = ix + incx
    end do

  end if

  return
end
subroutine qaxpy_complex ( n, sa, x, incx, y, incy )
  use kinds

!*****************************************************************************80
!
!! QAXPY adds a constant times one vector to another.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    08 April 1999
!
!  Author:
!
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
!    Algorithm 539: 
!    Basic Linear Algebra Subprograms for Fortran Usage,
!    ACM Transactions on Mathematical Software,
!    Volume 5, Number 3, September 1979, pages 308-323.
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the number of entries in the vector.
!
!    Input, real SA, the multiplier.
!
!    Input, real ( kind = qp ) X(*), the vector to be scaled and added to Y.
!
!    Input, integer ( kind = 4 ) INCX, the increment between successive entries of X.
!
!    Input/output, real ( kind = qp ) Y(*), the vector to which a 
!    multiple of X is to be added.
!
!    Input, integer ( kind = 4 ) INCY, the increment between successive entries of Y.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) i
  integer ( kind = 4 ) incx
  integer ( kind = 4 ) incy
  integer ( kind = 4 ) ix
  integer ( kind = 4 ) iy
  integer ( kind = 4 ) n
  complex ( kind = wp ) sa
  complex ( kind = wp ) x(*)
  complex ( kind = wp ) y(*)

  if ( n <= 0 ) then

  else if ( sa == 0.0Q+00 ) then

  else if ( incx == 1 .and. incy == 1 ) then

    y(1:n) = y(1:n) + sa * x(1:n)

  else

    if ( 0 <= incx ) then
      ix = 1
    else
      ix = ( - n + 1 ) * incx + 1
    end if

    if ( 0 <= incy ) then
      iy = 1
    else
      iy = ( - n + 1 ) * incy + 1
    end if

    do i = 1, n
      y(iy) = y(iy) + sa * x(ix)
      ix = ix + incx
      iy = iy + incy
    end do

  end if

  return
end
subroutine qgedi_complex ( a, lda, n, ipvt, det, work, job )
  use kinds

!*****************************************************************************80
!
!! QGEDI: determinant and inverse of a matrix factored by QGECO or QGEFA.
!
!  Discussion:
!
!    A division by zero will occur if the input factor contains
!    a zero on the diagonal and the inverse is requested.
!    It will not occur if the subroutines are called correctly
!    and if QGECO has set 0.0 < RCOND or QGEFA has set INFO == 0.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    15 March 2016
!
!  Author:
!
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!  Parameters:
!
!    Input/output, real ( kind = qp ) A(LDA,N), on input, the  LU factor
!    information, as output by QGECO or QGEFA.  On output, the inverse
!    matrix, if requested.
!
!    Input, integer ( kind = 4 ) LDA, the leading dimension of the array A.
!
!    Input, integer ( kind = 4 ) N, the order of the matrix A.
!
!    Input, integer ( kind = 4 ) IPVT(N), the pivot vector from QGECO or QGEFA.
!
!    Workspace, real ( kind = qp ) WORK(N).
!
!    Output, real ( kind = qp ) DET(2), the determinant of original matrix if
!    requested.  The determinant = DET(1) * 10.0**DET(2)
!    with  1.0 <= abs ( DET(1) ) < 10.0
!    or DET(1) == 0.0.
!
!    Input, integer ( kind = 4 ) JOB, specifies what is to be computed.
!    11, both determinant and inverse.
!    01, inverse only.
!    10, determinant only.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) lda
  integer ( kind = 4 ) n

  complex ( kind = wp ) a(lda,n)
  complex ( kind = wp ) det(2)
  integer ( kind = 4 ) i
  integer ( kind = 4 ) ipvt(n)
  integer ( kind = 4 ) j
  integer ( kind = 4 ) job
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  complex ( kind = wp ) t
  complex ( kind = wp ) work(n)
!
!  Compute the determinant.
!
  if ( job / 10 /= 0 ) then

    det(1) = 1.0Q+00
    det(2) = 0.0Q+00

    do i = 1, n

      if ( ipvt(i) /= i ) then
        det(1) = -det(1)
      end if

      det(1) = det(1) * a(i,i)

      if ( det(1) == 0.0Q+00 ) then
        exit
      end if

      do while ( abs ( det(1) ) < 1.0Q+00 )
        det(1) = det(1) * 10.0Q+00
        det(2) = det(2) - 1.0Q+00
      end do

      do while ( 10.0Q+00 <= abs ( det(1) ) )
        det(1) = det(1) / 10.0Q+00
        det(2) = det(2) + 1.0Q+00
      end do

    end do

  end if
!
!  Compute inverse(U).
!
  if ( mod ( job, 10 ) /= 0 ) then

    do k = 1, n

      a(k,k) = 1.0Q+00 / a(k,k)
      t = - a(k,k)
      call qscal_complex ( k-1, t, a(1,k), 1 )

      do j = k + 1, n
        t = a(k,j)
        a(k,j) = 0.0Q+00
        call qaxpy_complex ( k, t, a(1,k), 1, a(1,j), 1 )
      end do

    end do
!
!  Form inverse(U) * inverse(L).
!
    do k = n - 1, 1, -1

      work(k+1:n) = a(k+1:n,k)

      a(k+1:n,k) = 0.0Q+00

      do j = k + 1, n
        t = work(j)
        call qaxpy_complex ( n, t, a(1,j), 1, a(1,k), 1 )
      end do

      l = ipvt(k)
      if ( l /= k ) then
        call qswap_complex ( n, a(1,k), 1, a(1,l), 1 )
      end if

    end do

  end if

  return
end
subroutine qgefa_complex ( a, lda, n, ipvt, info )
  use kinds

!*****************************************************************************80
!
!! QGEFA factors a real matrix.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    07 March 2001
!
!  Author:
!
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!  Parameters:
!
!    Input/output, real ( kind = qp ) A(LDA,N).
!    On intput, the matrix to be factored.
!    On output, an upper triangular matrix and the multipliers used to obtain
!    it.  The factorization can be written A=L*U, where L is a product of
!    permutation and unit lower triangular matrices, and U is upper triangular.
!
!    Input, integer ( kind = 4 ) LDA, the leading dimension of A.
!
!    Input, integer ( kind = 4 ) N, the order of the matrix A.
!
!    Output, integer ( kind = 4 ) IPVT(N), the pivot indices.
!
!    Output, integer ( kind = 4 ) INFO, singularity indicator.
!    0, normal value.
!    K, if U(K,K) == 0.  This is not an error condition for this subroutine,
!    but it does indicate that QGESL or QGEDI will divide by zero if called.
!    Use RCOND in QGECO for a reliable indication of singularity.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) lda
  integer ( kind = 4 ) n

  complex ( kind = wp ) a(lda,n)
  integer ( kind = 4 ) info
  integer ( kind = 4 ) ipvt(n)
  !integer ( kind = 4 ) iqamax
  integer ( kind = 4 ) j
  integer ( kind = 4 ) k
  integer ( kind = 4 ) l
  complex ( kind = wp ) t
!
!  Gaussian elimination with partial pivoting.
!
  info = 0

  do k = 1, n - 1
!
!  Find L = pivot index.
!
    l = iqamax_complex ( n - k + 1, a(k,k), 1 ) + k - 1
    ipvt(k) = l
!
!  Zero pivot implies this column already triangularized.
!
    if ( a(l,k) == 0.0Q+00 ) then
      info = k
      cycle
    end if
!
!  Interchange if necessary.
!
    if ( l /= k ) then
      t      = a(l,k)
      a(l,k) = a(k,k)
      a(k,k) = t
    end if
!
!  Compute multipliers.
!
    a(k+1:n,k) = - a(k+1:n,k) / a(k,k)
!
!  Row elimination with column indexing.
!
    do j = k + 1, n
      t = a(l,j)
      if ( l /= k ) then
        a(l,j) = a(k,j)
        a(k,j) = t
      end if
      call qaxpy_complex ( n - k, t, a(k+1,k), 1, a(k+1,j), 1 )
    end do

  end do

  ipvt(n) = n

  if ( a(n,n) == 0.0Q+00 ) then
    info = n
  end if

  return
end
subroutine qscal_complex ( n, sa, x, incx )
  use kinds

!*****************************************************************************80
!
!! QSCAL scales a vector by a constant.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license. 
!
!  Modified:
!
!    15 March 2016
!
!  Author:
!
!    Original FORTRAN77 version by Charles Lawson, Richard Hanson, 
!    David Kincaid, Fred Krogh.
!    FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
!    Algorithm 539, 
!    Basic Linear Algebra Subprograms for Fortran Usage,
!    ACM Transactions on Mathematical Software,
!    Volume 5, Number 3, September 1979, pages 308-323.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the number of entries in the vector.
!
!    Input, real ( kind = qp ) SA, the multiplier.
!
!    Input/output, real ( kind = qp ) X(*), the vector to be scaled.
!
!    Input, integer ( kind = 4 ) INCX, the increment between successive 
!    entries of X.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) i
  integer ( kind = 4 ) incx
  integer ( kind = 4 ) ix
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n
  complex ( kind = wp ) sa
  complex ( kind = wp ) x(*)

  if ( n <= 0 ) then

  else if ( incx == 1 ) then

    m = mod ( n, 5 )

    x(1:m) = sa * x(1:m)

    do i = m + 1, n, 5
      x(i)   = sa * x(i)
      x(i+1) = sa * x(i+1)
      x(i+2) = sa * x(i+2)
      x(i+3) = sa * x(i+3)
      x(i+4) = sa * x(i+4)
    end do

  else

    if ( 0 <= incx ) then
      ix = 1
    else
      ix = ( - n + 1 ) * incx + 1
    end if

    do i = 1, n
      x(ix) = sa * x(ix)
      ix = ix + incx
    end do

  end if

  return
end
subroutine qswap_complex ( n, x, incx, y, incy )
  use kinds

!*****************************************************************************80
!
!! QSWAP interchanges two vectors.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license. 
!
!  Modified:
!
!    15 March 2016
!
!  Author:
!
!    This FORTRAN90 version by John Burkardt.
!
!  Reference:
!
!    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
!    LINPACK User's Guide,
!    SIAM, 1979,
!    ISBN13: 978-0-898711-72-1,
!    LC: QA214.L56.
!
!    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
!    Algorithm 539, 
!    Basic Linear Algebra Subprograms for Fortran Usage,
!    ACM Transactions on Mathematical Software, 
!    Volume 5, Number 3, September 1979, pages 308-323.
!
!  Parameters:
!
!    Input, integer ( kind = 4 ) N, the number of entries in the vectors.
!
!    Input/output, real ( kind = qp ) X(*), one of the vectors to swap.
!
!    Input, integer ( kind = 4 ) INCX, the increment between successive 
!    entries of X.
!
!    Input/output, real ( kind = qp ) Y(*), one of the vectors to swap.
!
!    Input, integer ( kind = 4 ) INCY, the increment between successive 
!    elements of Y.
!
  implicit none

!  integer, parameter :: qp = selected_real_kind ( 20, 199 )

  integer ( kind = 4 ) i
  integer ( kind = 4 ) incx
  integer ( kind = 4 ) incy
  integer ( kind = 4 ) ix
  integer ( kind = 4 ) iy
  integer ( kind = 4 ) m
  integer ( kind = 4 ) n
  complex ( kind = wp ) temp
  complex ( kind = wp ) x(*)
  complex ( kind = wp ) y(*)

  if ( n <= 0 ) then

  else if ( incx == 1 .and. incy == 1 ) then

    m = mod ( n, 3 )

    do i = 1, m
      temp = x(i)
      x(i) = y(i)
      y(i) = temp
    end do

    do i = m + 1, n, 3

      temp = x(i)
      x(i) = y(i)
      y(i) = temp

      temp = x(i+1)
      x(i+1) = y(i+1)
      y(i+1) = temp

      temp = x(i+2)
      x(i+2) = y(i+2)
      y(i+2) = temp

    end do

  else

    if ( 0 <= incx ) then
      ix = 1
    else
      ix = ( - n + 1 ) * incx + 1
    end if

    if ( 0 <= incy ) then
      iy = 1
    else
      iy = ( - n + 1 ) * incy + 1
    end if

    do i = 1, n
      temp = x(ix)
      x(ix) = y(iy)
      y(iy) = temp
      ix = ix + incx
      iy = iy + incy
    end do

  end if

  return
end

end module linpack_q_complex
