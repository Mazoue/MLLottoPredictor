namespace LottoPredictor
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using Encog.Neural.Networks;
    using Encog.ML.Data.Basic;
    using Encog.Neural.Networks.Layers;
    using Encog.Engine.Network.Activation;
    using Encog.Neural.Networks.Training.Propagation.Resilient;
    class Result
    {
        public int V1 { get; private set; }
        public int V2 { get; private set; }
        public int V3 { get; private set; }
        public int V4 { get; private set; }
        public int V5 { get; private set; }
        public int V6 { get; private set; }
        public int V7 { get; private set; }
        public Result(
            int v1, int v2, int v3, int v4, int v5, int v6, int v7)
        {
            V1 = v1; V2 = v2;
            V3 = v3; V4 = v4;
            V5 = v5;
            V6 = v6; V7 = v7;
        }
        public Result(double[] values)
        {
            V1 = (int)Math.Round(values[0]);
            V2 = (int)Math.Round(values[1]);
            V3 = (int)Math.Round(values[2]);
            V4 = (int)Math.Round(values[3]);
            V5 = (int)Math.Round(values[4]);
            V6 = (int)Math.Round(values[5]);
            V7 = (int)Math.Round(values[6]);
        }
        public bool IsValid()
        {
            return
            V1 >= 1 && V1 <= 50 &&
            V2 >= 1 && V2 <= 50 &&
            V3 >= 1 && V3 <= 50 &&
            V4 >= 1 && V4 <= 50 &&
            V5 >= 1 && V5 <= 50 &&
            V6 >= 1 && V6 <= 10 &&
            V7 >= 1 && V7 <= 10 &&
            V1 != V2 &&
            V1 != V3 &&
            V1 != V4 &&
            V1 != V5 &&
            V2 != V3 &&
            V2 != V4 &&
            V2 != V5 &&
            V3 != V4 &&
            V3 != V5 &&
            V4 != V5 &&
            V6 != V7;
        }
        public bool IsOut()
        {
            return
            !(V1 >= 1 && V1 <= 50 &&
              V2 >= 1 && V2 <= 50 &&
              V3 >= 1 && V3 <= 50 &&
              V4 >= 1 && V4 <= 50 &&
              V5 >= 1 && V5 <= 50 &&
              V6 >= 1 && V6 <= 10 &&
              V7 >= 1 && V7 <= 10);
        }
        public override string ToString()
        {
            return string.Format(
            "{0},{1},{2},{3},{4},{5},{6}",
            V1, V2, V3, V4, V5, V6, V7);
        }
    }
    class ListResults : List<Result> { }
    class Program
    {
        static void Main(string[] args)
        {
            var count = 0;
            var countMax = args.Length == 1 ? int.Parse(args[0]) : 27;
            var fileDB = "D:\\Temp\\LottoNumbers.txt";
            try
            {
                ListResults dbl = null;
                if (CreateDatabases(fileDB, out dbl))
                {
                    var deep = countMax;
                    var network = new BasicNetwork();
                    network.AddLayer(
                    new BasicLayer(null, true, 7 * deep));
                    network.AddLayer(
                    new BasicLayer(
                    new ActivationSigmoid(), true, 5 * 7 * deep));
                    network.AddLayer(
                    new BasicLayer(
                    new ActivationSigmoid(), true, 5 * 7 * deep));
                    network.AddLayer(
                    new BasicLayer(new ActivationLinear(), true, 7));
                    network.Structure.FinalizeStructure();
                    var learningInput = new double[deep][];
                    for (int i = 0; i < deep; ++i)
                    {
                        learningInput[i] = new double[deep * 7];
                        for (int j = 0, k = 0; j < deep; ++j)
                        {
                            var idx = 2 * deep - i - j;
                            var data = dbl[idx];
                            learningInput[i][k++] = data.V1;
                            learningInput[i][k++] = data.V2;
                            learningInput[i][k++] = data.V3;
                            learningInput[i][k++] = data.V4;
                            learningInput[i][k++] = data.V5;
                            learningInput[i][k++] = data.V6;
                            learningInput[i][k++] = data.V7;
                        }
                    }
                    var learningOutput = new double[deep][];
                    for (int i = 0; i < deep; ++i)
                    {
                        var idx = deep - 1 - i;
                        var data = dbl[idx];
                        learningOutput[i] = new double[7]
                        {
       data.V1,
       data.V2,
       data.V3,
       data.V4,
       data.V5,
       data.V6,
       data.V7,
                        };
                    }
                    var trainingSet = new BasicMLDataSet(
                    learningInput,
                    learningOutput);
                    var train = new ResilientPropagation(
                    network, trainingSet);
                    train.NumThreads = Environment.ProcessorCount;
                START:
                    System.Threading.Thread.Sleep(new Random().Next(10, 100));
                    network.Reset();
                RETRY:
                    var step = 0;
                    do
                    {
                        train.Iteration();
                        ++step;
                    }
                    while (train.Error > (0.0001 * countMax) && step < countMax);
                    var passedCount = 0;
                    for (var i = 0; i < deep; ++i)
                    {
                        var should =
                        new Result(learningOutput[i]);
                        var inputn = new BasicMLData(7 * deep);
                        Array.Copy(
                        learningInput[i],
                        inputn.Data,
                        inputn.Data.Length);
                        var comput =
                        new Result(
                        ((BasicMLData)network.
                        Compute(inputn)).Data);
                        var passed = should.ToString() == comput.ToString();
                        if (passed)
                        {
                            ++passedCount;
                        }
                    }
                    var input = new BasicMLData(7 * deep);
                    for (int i = 0, k = 0; i < deep; ++i)
                    {
                        var idx = deep - 1 - i;
                        var data = dbl[idx];
                        input.Data[k++] = data.V1;
                        input.Data[k++] = data.V2;
                        input.Data[k++] = data.V3;
                        input.Data[k++] = data.V4;
                        input.Data[k++] = data.V5;
                        input.Data[k++] = data.V6;
                        input.Data[k++] = data.V7;
                    }
                    var perfect = dbl[0];
                    var predict = new Result(
                    ((BasicMLData)network.Compute(input)).Data);
                    Console.ResetColor();
                    if (predict.IsOut())
                        goto START;
                    if (passedCount < (deep * 9d / 10) ||
                        !predict.IsValid())
                        goto RETRY;
                    if (PrintPredicts(predict, ++count, --countMax))
                        goto START;
                }
            }
            catch (Exception ex) { Console.WriteLine(ex.ToString()); }
            Console.WriteLine("FINISH");
        }
        static bool  CreateDatabases(
        string fileDB,
        out ListResults dbl)
        {
            dbl = new ListResults();
            using (var reader = File.OpenText(fileDB))
            {
                var line = string.Empty;
                var separator = new string[] { "," };
                while ((line = reader.ReadLine()) != null)
                {
                    var values = line.Split(separator, StringSplitOptions.None);
                    var ballOne = int.Parse(values[0]);
                    var ballTwo = int.Parse(values[1]);
                    var ballThree = int.Parse(values[2]);
                    var ballFour = int.Parse(values[3]);
                    var ballFive = int.Parse(values[4]);
                    var ballSix = int.Parse(values[5]);
                    var bonusBall = int.Parse(values[6]);
                    var powerBall = int.Parse(values[7]);


                    var res = new Result(
                    int.Parse(values[0]), int.Parse(values[1]),
                    int.Parse(values[2]), int.Parse(values[3]),
                    int.Parse(values[4]), int.Parse(values[6]),
                    int.Parse(values[7]));
                    dbl.Add(res);
                }
            }
            return true;
        }
        static IDictionary<int, int> vs = new Dictionary<int, int>(50);
        static IDictionary<int, int> cs = new Dictionary<int, int>(10);
        static bool PrintPredicts(Result r, int count = 1, int countMax = 0)
        {
            if (!vs.ContainsKey(r.V1)) vs.Add(r.V1, 1); else vs[r.V1]++;
            if (!vs.ContainsKey(r.V2)) vs.Add(r.V2, 1); else vs[r.V2]++;
            if (!vs.ContainsKey(r.V3)) vs.Add(r.V3, 1); else vs[r.V3]++;
            if (!vs.ContainsKey(r.V4)) vs.Add(r.V4, 1); else vs[r.V4]++;
            if (!vs.ContainsKey(r.V5)) vs.Add(r.V5, 1); else vs[r.V5]++;
            if (!cs.ContainsKey(r.V6)) cs.Add(r.V6, 1); else cs[r.V6]++;
            if (!cs.ContainsKey(r.V7)) cs.Add(r.V7, 1); else cs[r.V7]++;
            var vss = from p in vs
                      orderby p.Value descending, p.Key ascending
                      select string.Format("{0}:{1}%", p.Key, Math.Round(100m * p.Value / count, 2));
            var vtp =
            (from p in vs orderby p.Value descending select p).Take(5).Last().Value;
            var css = from p in cs
                      orderby p.Value descending, p.Key ascending
                      select string.Format("{0}:{1}%", p.Key, Math.Round(100m * p.Value / count, 2));
            var ctp =
            (from p in cs orderby p.Value descending select p).Take(2).Last().Value;
            Console.WriteLine("Try {0}: ({1} + {2})", count.ToString().PadLeft(2, '0'),
            string.Join(" ",
            from p in vs
            where p.Value >= vtp
            orderby p.Key ascending
            select p.Key),
            string.Join(" ",
            from p in cs where p.Value >= ctp orderby p.Key ascending select p.Key));
            Console.WriteLine(string.Join(" ", vss) + "\n" + string.Join(" ", css));
            return countMax > 0;
        }
    }
}