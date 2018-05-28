
import com.cra.figaro.algorithm.factored.VariableElimination
import com.cra.figaro.algorithm.sampling._
import com.cra.figaro.language._
import com.cra.figaro.library.collection.Container
import com.cra.figaro.library.compound.If

object Test {

  def Context (i : Double, j : Double): Array[Element[Boolean]] = {
    val c_i = If(Flip(i), 'H', 'T')
    val c_j = If(Flip(j), 'H', 'T')
    Array(
      c_i === 'H',
      c_j === 'H',
      c_i === 'H' && c_j === 'H'
    )
  }

  def Experiment_1 (): Unit = {

    val A1 = 0.6
    val B1 = 0.5
    val A2 = 0.2
    val B2 = 0.3
    val A3 = 0.5

    val C1 = Context(A2, B1)
    val C2 = Context(A1, B1)
    val C3 = Context(A1, B2)
    val C4 = Context(A3, B2)

    val alg = MetropolisHastings(1000000, ProposalScheme.default,
      C1(0), C1(1), C1(2),
      C2(0), C2(1), C2(2),
      C3(0), C3(1), C3(2),
      C4(0), C4(1), C4(2))
    alg.start()

    val C1_A2_B1 = alg.probability(C1(2), true)
    val C1_A2 = alg.probability(C1(0), true)
    val C1_B1 = alg.probability(C1(1), true)

    val C2_A1_B1 = alg.probability(C2(2), true)
    val C2_A1 = alg.probability(C2(0), true)
    val C2_B1 = alg.probability(C2(1), true)

    val C3_A1_B2 = alg.probability(C3(2), true)
    val C3_A1 = alg.probability(C3(0), true)
    val C3_B2 = alg.probability(C3(1), true)

    val C4_A3_B2 = alg.probability(C4(2), true)
    val C4_A3 = alg.probability(C4(0), true)
    val C4_B2 = alg.probability(C4(1), true)


    val A1_a = (C2_A1 + C3_A1) / 2
    val B1_a = (C1_B1 + C2_B1) / 2
    val B2_a = (C3_B2 + C4_B2) / 2

    println("A1 = " + A1_a)
    println("B1 = " + B1_a)
    println("B2 = " + B2_a)

    println("A2_B1 = " + C1_A2_B1)
    println("A2 = " + C1_A2)
    println("B1 = " + C1_B1)

    println("A1_B1 = " + C2_A1_B1)
    println("A1 = " + C2_A1)
    println("B1 = " + C2_B1)

    println("A1_B2 = " + C3_A1_B2)
    println("A1 = " + C3_A1)
    println("B2 = " + C3_B2)

    println("A3_B2 = " + C4_A3_B2)
    println("A3 = " + C4_A3)
    println("B2 = " + C4_B2)

    val LHS = ( A1_a * C1_A2 * B1_a * B2_a * C4_A3 )
    0.03325133 / 0.01257
    val RHS = ( C1_A2_B1 * C2_A1_B1 * C3_A1_B2 * C4_A3_B2 ) / ( A1_a * B1_a * B2_a )
    println(LHS)
    println(RHS)
  }

  def main(args: Array[String]) {

    Experiment_1()

  }
}