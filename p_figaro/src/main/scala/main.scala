import com.cra.figaro.algorithm.sampling._
import com.cra.figaro.language._
import com.cra.figaro.library.compound.{FastIf, If}

object Test {

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  val ch = 97
  def PTables (rvs : Array[Double],itr : Int): Array[Double] = {
    var a_c = Array[FastIf[Char]]()
    for( x <- rvs ){
      a_c :+= If(Flip(x), (0+ch).toChar, (1+ch).toChar)
    }
    var b_c = Array[Element[Boolean]]()
    for (y <- 0 to 1) {
      for (z <- 2 to 3) {
        var c_c = Array[Element[Boolean]]()
        for (a <- 0 to 1) {
          for (b <- 0 to 1) {
            c_c :+= (a_c(y) === (a+ch).toChar && a_c(z) === (b+ch).toChar)
          }
        }
        b_c ++= c_c
      }
    }
    val alg = MetropolisHastings(itr, ProposalScheme.default,b_c : _*)
    alg.start()
    var p = Array[Double]()
    for (i <- 0 to 15) {
      p :+= alg.probability(b_c(i), true)
    }
    p
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  def no_signalling_test(val1: Double, val2: Double,val3: Double,val4: Double): Unit = {
    if ((val1 + val2) == (val3 + val4)) {
      println("No signalling condition passed!")
    } else {
      println("No signalling condition failed!")
    }
  }

  def main(args: Array[String]) {

    val p = PTables(Array[Double](0.6, 0.2, 0.5, 0.3), 100000)

    println(p.deep.mkString("\n"))

    no_signalling_test(p(0),p(1),p(4),p(5))
    no_signalling_test(p(8),p(9),p(12),p(13))
    no_signalling_test(p(0),p(2),p(8),p(10))
    no_signalling_test(p(4),p(6),p(12),p(14))

    val A11 = (2 * (p(0) + p(1))) - 1
    val A12 = (2 * (p(4) + p(5))) - 1
    val A21 = (2 * (p(8) + p(9))) - 1
    val A22 = (2 * (p(12) + p(13))) - 1
    val B11 = (2 * (p(0) + p(2))) - 1
    val B12 = (2 * (p(8) + p(10))) - 1
    val B21 = (2 * (p(5) + p(6))) - 1
    val B22 = (2 * (p(12) + p(14))) - 1

    val Δ = (
            (Math.abs(A11) - Math.abs(A12))
            + (Math.abs(A21) - Math.abs(A22))
            + (Math.abs(B11) - Math.abs(B12))
            + (Math.abs(B21) - Math.abs(A22))) / 2

    if (Δ >= 1) {
      println("Delta is greater than or equal to 1")
    } else {
      println("Delta is smaller than 1 so contextuality may probably occur")
    }

    val A11B11 = (p(0) + p(3)) - (p(1) + p(2))
    val A12B12 = (p(4) + p(7)) - (p(5) + p(6))
    val A21B21 = (p(8) + p(11)) - (p(9) + p(10))
    val A22B22 = (p(12) + p(15)) - (p(13) + p(14))

    if ((A11B11 + A12B12 + A21B21 - A22B22) <= 2*(1+Δ)) {
      println("Bell scenario test 1 passed")
    } else {
      println("Bell scenario test 1 failed")
    }

    if ((A11B11 + A12B12 - A21B21 + A22B22) <= 2*(1+Δ)) {
      println("Bell scenario test 2 passed")
    } else {
      println("Bell scenario test 2 failed")
    }

    if ((A11B11 - A12B12 + A21B21 + A22B22) <= 2*(1+Δ)) {
      println("Bell scenario test 3 passed")
    } else {
      println("Bell scenario test 3 failed")
    }

    if ((0 - A11B11 + A12B12 - A21B21 + A22B22) <= 2*(1+Δ)) {
      println("Bell scenario test 4 passed")
    } else {
      println("Bell scenario test 4 failed")
    }

  }
}