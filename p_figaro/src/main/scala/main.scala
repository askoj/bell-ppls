:require ~/figaro-5.0.0.0-2.11/src/figaro_2.11-5.0.0.0-sources.jar

import com.cra.figaro.algorithm.sampling._
import com.cra.figaro.language._
import com.cra.figaro.library.compound.If
import com.cra.figaro.library.atomic.continuous.Uniform

object Main {
  def FoulisRandallProduct() : Array[Array[Array[Double]]] = {
    var foulisRandallEdges = Array[Array[Array[Double]]]()
    val hypergraphs = Array (
        Array ( Array ( Array (0.0 ,0.0) , Array (1.0 ,0.0) ) ,
        Array ( Array (0.0 ,1.0) , Array (1.0 ,1.0) ) ) ,
        Array ( Array ( Array (0.0 ,0.0) , Array (1.0 ,0.0) ) ,
        Array ( Array (0.0 ,1.0) , Array (1.0 ,1.0) ) ) )
    for (edgeA <- hypergraphs(0)) {
      for (edgeB <- hypergraphs(1)) {
        var foulisRandallEdge = Array[Array[Double]]()
        for (vertexA <- edgeA) {
          for (vertexB <- edgeB) {
            foulisRandallEdge ++= Array(Array [Double](
              vertexA (0) , vertexB (0) , vertexA (1) , vertexB (1) ) )
          } }
        foulisRandallEdges ++= Array(foulisRandallEdge)
      }
    }
    for (measurementChoice <- 0 to 1) {
      val measurementChoiceInverse = 1 - measurementChoice
      for ( edge <- hypergraphs ( measurementChoice ) ) {
        for (j <- 0 to 1) {
          var foulisRandallEdge = Array[Array[Double]]()
            for ( i <- edge.indices) {
              val edgeB = hypergraphs(measurementChoiceInverse)(i)
              val vertexA = edge(Math.abs(i - j))
              val vertexB = edgeB(0)
              val vertexC = edgeB(1)
              val verticesA = Array( vertexA(0) , vertexB( 0) , vertexA (1) , vertexB (1) )
              val verticesB = Array( vertexA(0) , vertexC( 0) , vertexA (1) , vertexC (1) )
              foulisRandallEdge ++= Array( Array[Double] (
                verticesA(measurementChoice) , verticesA ( measurementChoiceInverse ) ,
                verticesA ( measurementChoice+2) ,
                verticesA ( measurementChoiceInverse+2) ) )
              foulisRandallEdge ++= Array( Array[Double] (
               verticesB(measurementChoice) , verticesB ( measurementChoiceInverse ) ,
                verticesB(measurementChoice+2) ,
                verticesB ( measurementChoiceInverse+2) ) )
          }
          foulisRandallEdges ++= Array(foulisRandallEdge)
        }
      }
    }
    foulisRandallEdges
  }

  class Model() {
    var outcomes = Array[Element[Double]]()
    for (i <- 0 to 3) {
      outcomes :+= If(Flip(0.5), 0.0 , 1.0)
    }
    outcomes :+= Uniform (0.0 , 1.0)
  }
  def GetVertex(a: Int, b: Int, x: Int, y: Int): Int = { ((x*8)+(y*4))+(b+(a*2))
  }
  def GetHyperedges(H: Array[Array[Array[Double]]], n: Array[Double]) : Array[Int] = {
    var l = Array[Int]()
    for (i <- H.indices) {
      if (H(i).deep.contains(n.deep)) {
        l :+= i
      }
    }
    l
  }
  def GenerateGlobalDistribution( constraints : Array[Array[Array[Array[Double]]]], N: Int): Array[Double] = {
    val hyperedges = FoulisRandallProduct()
    var hyperedgesTallies = Array.fill(12){0.0}
    var globalDistribution = Array.fill(16){0.0}
    while (globalDistribution.sum < N) {
      var model = new Model()
      val algorithm = MetropolisHastings(N, ProposalScheme.default , model.outcomes: _*)
      algorithm.start()
      algorithm.stop()
      algorithm.kill()
      val a = algorithm.sampleFromPosterior( model . outcomes(0) ) . take (N) . toArray
      val b = algorithm . sampleFromPosterior ( model . outcomes(1) ) . take (N) . toArray
      val x = algorithm . sampleFromPosterior ( model . outcomes(2) ) . take (N) . toArray
      val y = algorithm . sampleFromPosterior ( model . outcomes(3) ) . take (N) . toArray
      val c = algorithm . sampleFromPosterior ( model . outcomes(4) ) . take (N) . toArray
      for (i <- 0 until N) {
        val x_x = x(i).toInt
        val y_y = y(i).toInt
        val a_a = a(i).toInt

        val b_b = b(i).toInt
        if (c(i) < constraints(x_x)(y_y)(a_a)(b_b)) {

          for ( edge <- GetHyperedges(hyperedges, Array(a_a, b_b, x_x, y_y))) {
            hyperedgesTallies(edge) += 1.0
          }
          globalDistribution(
            GetVertex(a_a, b_b, x_x, y_y)) += 1.0 }
      } }
    for (a <- 0 to 1; b <- 0 to 1; x <- 0 to 1; y <- 0 to 1) { var summedAmount = 0.0
      val associatedHyperedges = GetHyperedges ( hyperedges ,
        Array(a.toDouble ,b.toDouble ,x.toDouble ,y.toDouble) )
      for (edgeIndex <- associatedHyperedges.indices) {
        summedAmount += hyperedgesTallies ( edgeIndex )
      }
      globalDistribution(GetVertex(a,b,x,y)) = globalDistribution(GetVertex(a,b,x,y)) / summedAmount
    }
    globalDistribution
  }

  def main(args: Array[String]): Unit = {

    val constraints = Array(
      Array( Array(Array(0.5, 0.0), Array(0.0, 0.5)), Array(Array(0.5, 0.0), Array(0.0, 0.5)) ),
      Array( Array(Array(0.5, 0.0), Array(0.0, 0.5)), Array(Array(0.0, 0.5), Array(0.5, 0.0)) ) )

    println("Testing 1000")
    val result = time { GenerateGlobalDistribution(constraints,1000).deep }
    println(result)
    println("Testing 2000")
    val result = time { GenerateGlobalDistribution(constraints,2000).deep }
    println(result)
    println("Testing 3000")
    val result = time { GenerateGlobalDistribution(constraints,3000).deep }
    println(result)
    println("Testing 4000")
    val result = time { GenerateGlobalDistribution(constraints,4000).deep }
    println(result)
    println("Testing 5000")
    val result = time { GenerateGlobalDistribution(constraints,5000).deep }
    println(result)
    println("Testing 6000")
    val result = time { GenerateGlobalDistribution(constraints,6000).deep }
    println(result)
    println("Testing 7000")
    val result = time { GenerateGlobalDistribution(constraints,7000).deep }
    println(result)
    println("Testing 8000")
    val result = time { GenerateGlobalDistribution(constraints,8000).deep }
    println(result)
    println("Testing 9000")
    val result = time { GenerateGlobalDistribution(constraints,9000).deep }
    println(result)
    println("Testing 10000")
    val result = time { GenerateGlobalDistribution(constraints,10000).deep }
    println(result)
    println("Testing 15000")
    val result = time { GenerateGlobalDistribution(constraints,15000).deep }
    println(result)
    println("Testing 20000")
    val result = time { GenerateGlobalDistribution(constraints,20000).deep }
    println(result)
    println("Testing 25000")
    val result = time { GenerateGlobalDistribution(constraints,25000).deep }
    println(result)
    println("Testing 30000")
    val result = time { GenerateGlobalDistribution(constraints,30000).deep }
    println(result)
    println("Testing 35000")
    val result = time { GenerateGlobalDistribution(constraints,35000).deep }
    println(result)
    println("Testing 40000")
    val result = time { GenerateGlobalDistribution(constraints,40000).deep }
    println(result)
    println("Testing 45000")
    val result = time { GenerateGlobalDistribution(constraints,45000).deep }
    println(result)
    println("Testing 50000")
    val result = time { GenerateGlobalDistribution(constraints,50000).deep }
    println(result)
    println("Testing 55000")
    val result = time { GenerateGlobalDistribution(constraints,55000).deep }
    println(result)
    println("Testing 60000")
    val result = time { GenerateGlobalDistribution(constraints,60000).deep }
    println(result)
    println("Testing 65000")
    val result = time { GenerateGlobalDistribution(constraints,65000).deep }
    println(result)
    println("Testing 70000")
    val result = time { GenerateGlobalDistribution(constraints,70000).deep }
    println(result)
    println("Testing 75000")
    val result = time { GenerateGlobalDistribution(constraints,75000).deep }
    println(result)
    println("Testing 80000")
    val result = time { GenerateGlobalDistribution(constraints,80000).deep }
    println(result)
    println("Testing 85000")
    val result = time { GenerateGlobalDistribution(constraints,85000).deep }
    println(result)
    println("Testing 90000")
    val result = time { GenerateGlobalDistribution(constraints,90000).deep }
    println(result)
    println("Testing 95000")
    val result = time { GenerateGlobalDistribution(constraints,95000).deep }
    println(result)
    println("Testing 100000")
    val result = time { GenerateGlobalDistribution(constraints,100000).deep }
    println(result)
  }

}
