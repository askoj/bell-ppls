
import com.cra.figaro.algorithm.sampling._
import com.cra.figaro.language._
import com.cra.figaro.library.compound.If

object Test {

  val startTimeMillis = System.currentTimeMillis()

  class Model(Probability1 : Double, Probability2 : Double) {
    val V1 = If(Flip(Probability1), 1, 0)
    val V2 = If(Flip(Probability2), 1, 0)
    var JointOutcomes = Array[Element[Boolean]]()
    for (a <- 0 to 1; b <- 0 to 1)
      JointOutcomes :+= (V1 === a && V2 === b)
  }

  def Context(model : Model, iterations : Int): Array[Double] = {
    val algorithm = MetropolisHastings(iterations,
      ProposalScheme.default,model.JointOutcomes : _*)
    algorithm.start
    algorithm.stop
    var jointProbabilities = Array[Double]()
    for (i <- 0 to 3)
      jointProbabilities :+=
        algorithm.probability(model.JointOutcomes(i), true)
    algorithm.kill
    jointProbabilities
  }

  def main(args: Array[String]) {
    val A1 = 0.6
    val A2 = 0.2
    val B1 = 0.2
    val B2 = 0.2
    var globalDistribution = Array[Double]()
    globalDistribution ++= Context(new Model(A1, B1),50000)
    globalDistribution ++= Context(new Model(A1, B2),50000)
    globalDistribution ++= Context(new Model(A2, B1),50000)
    globalDistribution ++= Context(new Model(A2, B2),50000)
    val p = globalDistribution
    println(p.deep)

    def signalling(v1: Int, v2: Int, v3: Int, v4: Int): Boolean = {
      println((p(v1)+p(v2)).toString()+" = "+(p(v3)+p(v4)).toString())
      println((p(v1)+p(v2))-(p(v3)+p(v4)))
      Math.abs((p(v1)+p(v2))-(p(v3)+p(v4))) < 0.01
    }

    def equality(v1: Int, v2: Int, v3: Int, v4: Int): Boolean = {
      def f1(v1: Int, v2: Int): Double = {
        Math.abs((2 * (p(v1) + p(v2))) - 1)
      }
      def f2(v1: Int, v2: Int, v3: Int, v4: Int): Double = {
        (p(v1) + p(v2)) - (p(v3) + p(v4))
      }
      val delta = 0.5 * (
        (f1(0,1) - f1(4,5)) + (f1(8,9) - f1(12,13)) +
          (f1(0,2) - f1(4,6)) + (f1(8,10) - f1(12,14)))
      (2 * (1 + delta)) >= Math.abs(
        (v1*f2(0,3,1,2)) + (v2*f2(4,7,5,6)) +
          (v3*f2(8,11,9,10)) + (v4*f2(12,15,13,14)))
    }

    val tests = Array[Boolean](
      signalling(0,1,4,5),
      signalling(8,9,12,13),
      signalling(0,2,8,10),
      signalling(4,6,12,14),
      equality(1,1,1,-1),
      equality(1,1,-1,1),
      equality(1,-1,1,1),
      equality(-1,1,1,1)
    )

    println(tests.deep)

    val endTimeMillis = System.currentTimeMillis()
    val durationSeconds = (endTimeMillis - startTimeMillis) / 1000.0
    println(durationSeconds)
  }

}